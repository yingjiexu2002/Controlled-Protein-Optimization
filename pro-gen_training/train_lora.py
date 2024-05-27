from __future__ import print_function
from __future__ import division
import time
import sys
import torch
import os
import tqdm
import pdb
import numpy as np
import platform
import hashlib
# import pytorch_transformer
import transformer_lora as pytorch_transformer
from transformer_lora import lora_config
import loralib as lora
import re
import argparse
from torch.utils.tensorboard import SummaryWriter
from transformProtein_simplified import transformProtein
from ProteinDataset import ProteinDataset
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.distributed as dist  # 分布式训练


use_py3 = platform.python_version()[0] == '3'   # 检查当前运行的Python版本是否为3。如果是，则将变量use_py3设置为True，否则为False

#***************参数解析********************
parser = argparse.ArgumentParser(description='Finetuning code')
parser.add_argument('--model_dir', type=str, default='checkpoints/',
                    help='location of training model checkpoint')       # 模型输出路径
parser.add_argument('--model_path', type=str,
                    default='checkpoints\pretrain_progen_full.pth',
                    help='location of model *data* checkpoint to load; this is NOT the directory but rather the model checkpoint')  # 上次保存的checkpoint路径
parser.add_argument('--seed', type=int, default=313,
                    help='random seed for TensorFlow, numpy and PythonHash')    # 随机数种子
parser.add_argument('--sequence_len', type=int, default=511,
                    help='sequence len of model being fine-tuned')
parser.add_argument('--num_epochs', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--num_layers', type=int, default=36,
                    help='number of transfomer layers. used for loading checkpoint')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for dataloader')
parser.add_argument('--vocab_loc', type=str, default='mapping_files/vocab.txt', help='vocab location')
parser.add_argument('--num_workers', type=int, default=0, help='for dataloader')
parser.add_argument('--warmup_iteration', type=int, default=1000, help='LR warmup cutoff')
parser.add_argument('--save_iter', type=int, default=1000, help='save model checkpoint every X iterations')
parser.add_argument('--pklpath', type=str, default='dataset/', help='dataset pickle dir path')
parser.add_argument('--lora_dim', type=int, default= 4, help='lora intrinsic dimension')
parser.add_argument('--lora_alpha', type=int, default=32, help='lora alpha')
parser.add_argument('--lora_dropout', type=float, default=0.1)
parser.add_argument('--log_interval', type=int, default=100, help='log interval')
parser.add_argument('--learning_rate', type=float, default=0.0001)  # 固定学习率
parser.add_argument("--resume", action="store_true", help="resume from the latest checkpoint.")  # 从上一次训练的检查点继续训练
parser.add_argument('--local-rank', default=0, type=int, help='node rank for distributed training')  # 分布式训练
args = parser.parse_args()
#***************参数解析********************

# 分布式训练
print(os.environ['MASTER_ADDR'])
print(os.environ['MASTER_PORT'])
world_size = torch.cuda.device_count()  # GPU数量
local_rank = args.local_rank  # 进程序列号
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)

# 设置随机数种子
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)

# load the vocabulary from file
vocab = open(args.vocab_loc).readlines() if not use_py3 else open(args.vocab_loc, encoding='utf-8').read().split('\n')[:-1]
vocab = list(map(lambda x: x.split(' ')[0], vocab))

# length of the vocabulary
vocab_size = len(vocab)
print('-----vocab size', vocab_size, '------')

# sequence length to use for transfomer
seq_length = args.sequence_len

embedding_dim = 1280

class TiedEmbeddingSoftmax(torch.nn.Module):
  '''
  实现一个绑定的 嵌入-softmax 模型
  '''
  def __init__(self, vocab_size=vocab_size, embedding_size=embedding_dim, **kwargs):
    """
    vocab_size表示词汇表的大小
    embedding_size表示嵌入向量的维度
    """
    super(TiedEmbeddingSoftmax, self).__init__()
    # 创建可训练权重
    self.w = torch.nn.Parameter(torch.normal(0., 1e-2, size=(vocab_size, embedding_size)))
    self.b = torch.nn.Parameter(torch.zeros(vocab_size))

  def forward(self, inputs, embed=True):
    if embed:
      return torch.nn.functional.embedding(inputs, self.w)
    else:
      # 进行计算softmax之前的操作，softmax操作在main函数中
      # 这个操作用于计算每个词的得分，可用于后续的softmax计算
      # ？？？？？？？？？？？？？？？为什么可以直接用embedding的权重w
      return torch.tensordot(inputs, self.w.t(), 1) + self.b

class CTRLmodel(torch.nn.Module):
  '''定义一个基于CTRL的模型'''
  def __init__(self, loraconfig):
    super(CTRLmodel,self).__init__()
    # 创建实例
    self.tied_embedding_softmax = TiedEmbeddingSoftmax()
    self.encoder = pytorch_transformer.Encoder(loraconfig=loraconfig)
  
  def forward(self, inputs):
    # 将输入序列转化为嵌入向量
    x = self.tied_embedding_softmax(inputs, embed=True)
    x = self.encoder(x)
    x = self.tied_embedding_softmax(x, embed=False)
    return x

  # 加载预训练的模型
  def loadCheckpoint(self, model_path, num_layers,pt=False):
    pytorch_model_hash = hashlib.md5(model_path.encode('utf-8')).hexdigest()

    if os.path.exists(pytorch_model_hash):
        print('Found PyTorch checkpoint @', pytorch_model_hash)
        print('Loading instead of converting from TensorFlow')
        checkpoint = torch.load(pytorch_model_hash)
        self.tied_embedding_softmax.load_state_dict(checkpoint['softmax'])
        self.encoder.load_state_dict(checkpoint['encoder'])

        self.tied_embedding_softmax.to('cuda')
        self.encoder.to('cuda')


# lora参数配置
l_config = lora_config(
    lora_attn_dim=args.lora_dim, 
    lora_attn_alpha=args.lora_alpha, 
    lora_dropout=args.lora_dropout,
)

# 加载模型
model = CTRLmodel(l_config)
print('model initialized')
if args.resume :
  model.load_state_dict(torch.load('checkpoints/lora/ProGen_finetuned_with_lora.pt')["model_state_dict"], strict=False)
  model.load_state_dict(torch.load('checkpoints/lora/lora_model.165000.pt')["model_state_dict"], strict=False)
  print("恢复上次训练的最终位置，继续训练")
else:
  model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu'))["model_state_dict"], strict=False)
  print('previous checkpoint loaded')
model = model.cuda()


# ---------------------训练参数调整--------------------------------
# 使用mark_only_lora_as_trainable包装lm_net，表示只对低秩矩阵做训练
if args.lora_dim > 0:
    lora.mark_only_lora_as_trainable(model)
    print("设置仅更新Lora的参数")
model.tied_embedding_softmax.w.requires_grad = True
model.tied_embedding_softmax.b.requires_grad = True

# 分布式训练
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],output_device=local_rank)  # 多卡训练

# 训练
class Trainer(object):
    def __init__(self, model, warmup_iteration, seq_length, batch_size, num_workers, vocab_size, model_dir, save_iter):
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_size = vocab_size
        self.model_dir = model_dir
        self.save_iter = save_iter
        self.firstAAidx = self.vocab_size - 26      # Assuming that the pad token is the last token and AAs are at the end
                                                    # 查看mapping_files/vocab.txt，倒数第26个为氨基酸编码的起始位置

        # 使用adam优化器来进行梯度下降计算，并传入所有的可学习参数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)  # lr, betas

        # ---------------------------------策略一、定义lambda函数，‘iteration’表示当前训练步数，用于warmup动态调整学习率
        # lambdafn = lambda iteration: min(iteration / (warmup_iteration * 1.0), 1.0)
        # # 定义学习率调度器
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambdafn)
        # ---------------------------------策略二、静态学习率
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        # ---------------------------------策略三、等间隔调整学习率
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.warmup_iteration, gamma=0.5)

        # 极大似然估计与最小化交叉熵损失等价，详细可参考：https://zhuanlan.zhihu.com/p/84764177
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocab_size - 1, reduction='none')

        self.transformFull = transformProtein(maxSampleLength=seq_length + 1,
                                              selectSwiss=1.0, selectTrembl=1.0,
                                              maxTaxaPerSample=3, maxKwPerSample=5, dropRate=0.0)
        self.transformPartial = transformProtein(maxSampleLength=seq_length + 1,
                                                 selectSwiss=0.9, selectTrembl=0.9,
                                                 maxTaxaPerSample=3, maxKwPerSample=5, dropRate=0.2)
        self.transformNone = transformProtein(maxSampleLength=seq_length + 1,
                                              selectSwiss=1.0, selectTrembl=1.0,
                                              maxTaxaPerSample=0, maxKwPerSample=0, dropRate=1.0)

        self.writer = SummaryWriter("logs/log1")   # from tensorboard

    def train(self, num_epochs):
        self.model.train()  # 设置为训练模式

        if local_rank == 0:
          train_print_path = os.path.join(args.model_dir, 'print_result.txt')
        # with open(train_print_path, "w") as file:
        #     file.truncate(0)    # 清空之前输出的文件内容
        iter_num = 0
        for epoch in range(num_epochs):
            if local_rank == 0:
              current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
              prt_str = f"第{epoch}个epoch，开始时间为：{current_time}"
              print(prt_str)
              with open(train_print_path, "a") as file:
                file.write(prt_str)
                file.write('\n')

            loss_e = 0.0
            num_e = 0
            pklpath = args.pklpath
            # 加载数据块
            print("加载路径为："+ str(pklpath)+" 的数据块")
            chunk_dataset = ProteinDataset(pklpath, firstAAidx=self.firstAAidx, transformFull=self.transformFull,
                                            transformPartial=self.transformPartial, transformNone=self.transformNone)
            # # 将数据块数据分批加载到GPU中进行训练
            # dataloader = DataLoader(chunk_dataset, shuffle=True, batch_size=self.batch_size,
            #                         num_workers=self.num_workers, pin_memory=False)  # TODO pinmem?
            

            # 对训练数据集做修改。将dataloader的sampler修改为DistributedSampler，这样保证其每个进程采样的数据是不同的
            train_sampler = torch.utils.data.distributed.DistributedSampler(chunk_dataset)
            # 将数据块数据分批加载到GPU中进行训练，分布式训练时shuffle=False
            dataloader = DataLoader(chunk_dataset, shuffle=False, batch_size=self.batch_size,sampler=train_sampler,
                                    num_workers=self.num_workers, pin_memory=False)  # TODO pinmem?

            for i, (sample, labels, existence, padIndex, begAAindex) in enumerate(dataloader):
                log_start_time = time.time()
                self.optimizer.zero_grad()  # 将优化器的梯度清零

                sample, labels, existence, padIndex = sample.cuda(), labels.cuda(), existence.cuda(), padIndex.cuda()   # 将相关数据转移到GPU上
                output = self.model(sample)

                loss = self.criterion(output.permute(0, 2, 1), labels)  # 计算模型输出与真实标签的损失值
                loss = torch.mean((torch.sum(loss, dim=1) / padIndex) * existence)  # pad masking, loss weighting
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)   # 裁剪梯度
                self.optimizer.step()   # 更新参数
                self.scheduler.step()   # 更新学习率
                loss_e += loss.item()   # 累积每个epoch的总损失值和样本数量
                num_e += sample.shape[0]
                iter_num += 1
                if local_rank == 0:
                  self.writer.add_scalar('Loss_iteration', loss.item(), iter_num)     # 利用tensorboard记录损失变化情况

                # 打印训练日志
                if (iter_num % args.log_interval == 0) and local_rank == 0: 
                    elapsed = time.time() - log_start_time
                    lr = self.optimizer.param_groups[0]['lr']
                    log_str = f'| epoch {epoch:3d} step {iter_num:>8d} | ' \
                            f'lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | ' \
                            f'loss {loss:5.2f} | average loss {loss_e / num_e:5.2f} | '
                    print(log_str)
                    with open(train_print_path, "a") as file:
                        file.write(log_str)
                        file.write('\n')
                    log_start_time = time.time()

                # # 保存模型
                # if (iter_num) % self.save_iter == 0:
                #     lora_model_path = os.path.join(args.model_dir, f'lora_model.{iter_num}.pt')
                #     print('saving checkpoint', lora_model_path)
                #     # 调用lora的lora_state_dict，在存储的时候只会保存lora部分的权重
                #     torch.save({'model_state_dict': lora.lora_state_dict(model)}, lora_model_path)
            loss_e /= num_e     # 计算每个epoch的平均损失值
    
        if local_rank==0:
          lora_model_path = os.path.join(args.model_dir, f'lora_model.{iter_num}.pt')
          print('saving checkpoint', lora_model_path)
          # 调用lora的lora_state_dict，在存储的时候只会保存lora部分的权重
          torch.save({'model_state_dict': lora.lora_state_dict(model.module)}, lora_model_path) # 分布式训练，需要改为model.module

          print("保存最终的微调主模型")
          checkpoint_path = os.path.join(args.model_dir, f'progen_finetuned_with_lora.pt')
          torch.save({'model_state_dict': self.model.module.state_dict()}, checkpoint_path)  # 分布式训练，需要改为model.module

training = Trainer(model=model, warmup_iteration=args.warmup_iteration, seq_length=seq_length,
                   batch_size=args.batch_size, num_workers=args.num_workers, vocab_size=vocab_size,
                   model_dir=args.model_dir, save_iter=args.save_iter)
print('begin training...')
training.train(args.num_epochs)