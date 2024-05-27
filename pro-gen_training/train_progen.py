from __future__ import print_function
from __future__ import division
import sys
import torch
import os
from tqdm import tqdm
import pdb
import numpy as np
import platform
import hashlib
import pytorch_transformer  # 使用基本的transformer
import re
import argparse
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformProtein_simplified import transformProtein
from ProteinDataset import ProteinDataset
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.distributed as dist  # 分布式训练

use_py3 = platform.python_version()[0] == '3'

parser = argparse.ArgumentParser(description='TensorFlow code for generating from CTRL')
parser.add_argument('--model_dir', type =str, default='model_v0.pth',
                                        help='location of training model checkpoint')
parser.add_argument('--model_path', type=str, default='/home/amadani/ctrl/ckpt/seqlen256_36layers_v0.ckpt/model.ckpt-684000', help='location of model *data* checkpoint to load; this is NOT the directory but rather the model checkpoint')
parser.add_argument('--seed', type=int, default=313,
                                        help='random seed for TensorFlow, numpy and PythonHash')
parser.add_argument('--sequence_len', type=int, default=511,
                                        help='sequence len of model being fine-tuned')
parser.add_argument('--num_epochs', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--num_layers', type=int, default=36, help='number of transfomer layers. used for loading checkpoint')
parser.add_argument('--batch_size', type=int, default = 4, help='batch size for dataloader')
parser.add_argument('--vocab_loc', type=str, default='mapping_files/vocab.txt', help='vocab location')
parser.add_argument('--num_workers', type=int, default=0, help='for dataloader')
parser.add_argument('--warmup_iteration', type=int, default=1000, help='LR warmup cutoff')
parser.add_argument('--save_iter', type=int, default=1000, help='save model checkpoint every X iterations')
parser.add_argument('--pklpath', type=str, default='dataset/', help='dataset pickle dir path')
parser.add_argument('--log_interval', type=int, default=100, help='log interval')
parser.add_argument('--learning_rate', type=float, default=0.0001)  # 固定学习率
parser.add_argument('--local-rank', default=0, type=int, help='node rank for distributed training')  # 分布式训练
args = parser.parse_args()

# 多卡训练
print(os.environ['MASTER_ADDR'])
print(os.environ['MASTER_PORT'])
world_size = torch.cuda.device_count()  # GPU数量
local_rank = args.local_rank  # 进程序列号
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)

# load the vocabulary from file
vocab = open(args.vocab_loc).readlines() if not use_py3 else open(args.vocab_loc, encoding='utf-8').read().split('\n')[:-1]
vocab = list(map(lambda x: x.split(' ')[0], vocab))
# length of the vocabulary
vocab_size = len(vocab)
print('-----vocab size',vocab_size,'------')

# sequence length to use for transfomer
seq_length = args.sequence_len

embedding_dim = 1280

class TiedEmbeddingSoftmax(torch.nn.Module):

  def __init__(self, vocab_size=vocab_size, embedding_size=embedding_dim, **kwargs):
    super(TiedEmbeddingSoftmax, self).__init__()
    self.w = torch.nn.Parameter(torch.normal(0., 1e-2, size=(vocab_size, embedding_size)))
    self.b = torch.nn.Parameter(torch.zeros(vocab_size))

  def forward(self, inputs, embed=True):
    if embed:
      return torch.nn.functional.embedding(inputs, self.w)
    else:
      return torch.tensordot(inputs, self.w.t(), 1) + self.b

class CTRLmodel(torch.nn.Module):
  def __init__(self):
    super(CTRLmodel,self).__init__()
    self.tied_embedding_softmax = TiedEmbeddingSoftmax()
    self.encoder = pytorch_transformer.Encoder()

  def forward(self, inputs):
    x = self.tied_embedding_softmax(inputs, embed = True)
    x = self.encoder(x)
    x = self.tied_embedding_softmax(x, embed = False)
    return x

  def loadCheckpoint(self, model_path, num_layers):
    # pytorch_model_hash = hashlib.md5(model_path.encode('utf-8')).hexdigest()
    pytorch_model_hash = model_path

    if os.path.exists(pytorch_model_hash):
      print('Found PyTorch checkpoint @', pytorch_model_hash)
      print('Loading instead of converting from TensorFlow')
      checkpoint = torch.load(pytorch_model_hash)
      self.tied_embedding_softmax.load_state_dict(checkpoint['softmax'])
      self.encoder.load_state_dict(checkpoint['encoder'])

      self.tied_embedding_softmax.to('cuda')
      self.encoder.to('cuda')

    else:
      print('Could not find PyTorch checkpoint')
      print('Converting weights and will store the PyTorch checkpoint as ', pytorch_model_hash)
      exit(0)

# 加载模型
model = CTRLmodel()
if local_rank == 0:
  print('model initialized')
model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu'))["model_state_dict"], strict=False)
if local_rank == 0:
  print('previous checkpoint loaded')
model = model.cuda()

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],output_device=local_rank)  # 多卡训练

# # freeze all weights except embedding
# for p in model.parameters():
#     p.requires_grad=False
# model.tied_embedding_softmax.w.requires_grad=True
# model.tied_embedding_softmax.b.requires_grad=True
print('全参数微调！')

class Trainer(object):
    def __init__(self, model, warmup_iteration, seq_length, batch_size, num_workers, vocab_size, model_dir, save_iter):
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_size = vocab_size
        self.model_dir = model_dir
        self.save_iter = save_iter
        self.firstAAidx = self.vocab_size - 26 # Assuming that the pad token is the last token and AAs are at the end
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate) #lr, betas
        # lambdafn = lambda iteration: min(iteration/(warmup_iteration*1.0),1.0)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambdafn)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0) # 静态学习率
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocab_size-1, reduction='none')
        
        self.transformFull = transformProtein(maxSampleLength=seq_length + 1,
                                              selectSwiss=1.0, selectTrembl=1.0,
                                              maxTaxaPerSample=3, maxKwPerSample=5, dropRate=0.0)
        self.writer = SummaryWriter("logs/log0")

    def train(self, num_epochs):
        self.model.train()

        iter_num = 0
        # 使用tqdm显示进度条
        for epoch in tqdm(range(num_epochs)):
            loss_e = 0.0
            num_e = 0

            pklpath = args.pklpath
            # chunk_dataset = ProteinDataset(pklpath, firstAAidx = self.firstAAidx, transformFull = self.transformFull)
            # dataloader = DataLoader(chunk_dataset, shuffle = True, batch_size = self.batch_size,
            #                         num_workers = self.num_workers, pin_memory = False) #TODO pinmem?

            # 对训练数据集做修改。将dataloader的sampler修改为DistributedSampler，这样保证其每个进程采样的数据是不同的
            chunk_dataset = ProteinDataset(pklpath, firstAAidx=self.firstAAidx, transformFull=self.transformFull)
            train_sampler = torch.utils.data.distributed.DistributedSampler(chunk_dataset)
            # 将数据块数据分批加载到GPU中进行训练，分布式训练时shuffle=False
            dataloader = DataLoader(chunk_dataset, shuffle=False, batch_size=self.batch_size,sampler=train_sampler,
                                    num_workers=self.num_workers, pin_memory=False)  # TODO pinmem?
            
            for i, (sample, labels, existence, padIndex, begAAindex) in enumerate(dataloader):
                # # 用作调试
                # torch.set_printoptions(threshold=np.inf)
                # print("sample:", sample)
                # exit(0)
                self.optimizer.zero_grad()
                sample, labels, existence, padIndex = sample.cuda(), labels.cuda(), existence.cuda(), padIndex.cuda()
                output = self.model(sample)
                #pdb.set_trace()
                loss = self.criterion(output.permute(0,2,1), labels)
                loss = torch.mean((torch.sum(loss,dim=1)/padIndex)*existence) #pad masking, loss weighting
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                self.optimizer.step()
                self.scheduler.step()
                loss_e += loss.item()
                num_e += sample.shape[0]
                iter_num += 1
                if local_rank == 0:
                  self.writer.add_scalar('Loss_iteration',loss.item(),iter_num)
  
            loss_e/=num_e
            #print(loss_e)
            #self.writer.add_scalar('Loss_epoch',loss_e, epoch)

            if local_rank==0:
              print("保存模型")
              checkpoint_path = os.path.join(args.model_dir, f'my_progen_finetuned.pt')
              torch.save({'model_state_dict': self.model.module.state_dict()}, checkpoint_path)

training = Trainer(model=model, warmup_iteration=args.warmup_iteration, seq_length=seq_length,
                   batch_size=args.batch_size, num_workers=args.num_workers, vocab_size=vocab_size,
                   model_dir = args.model_dir, save_iter=args.save_iter)
print('begin training...')
training.train(args.num_epochs)



