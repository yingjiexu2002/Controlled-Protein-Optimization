from __future__ import print_function
from __future__ import division
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import tqdm
import numpy as np
import platform
import hashlib
import fast_transformer as pytorch_transformer
import argparse
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import pickle
import time



class TiedEmbeddingSoftmax(torch.nn.Module):

  def __init__(self, vocab_size, embedding_size, **kwargs):
    super(TiedEmbeddingSoftmax, self).__init__()
    self.w = torch.nn.Parameter(torch.normal(0., 1e-2, size=(vocab_size, embedding_size)))
    self.b = torch.nn.Parameter(torch.zeros(vocab_size))

  def forward(self, inputs, embed=True):
    if embed:
      return torch.nn.functional.embedding(inputs, self.w)
    else:
      return torch.tensordot(inputs, self.w.t(), 1) + self.b

class CTRLmodel(torch.nn.Module):
  def __init__(self,vocab_size,embedding_size):
    super(CTRLmodel,self).__init__()
    self.tied_embedding_softmax = TiedEmbeddingSoftmax(vocab_size, embedding_size)
    self.encoder = pytorch_transformer.Encoder()


  def forward(self, inputs, past=None):
    x = self.tied_embedding_softmax(inputs, embed = True)
    x, past = self.encoder(x, past)
    x = self.tied_embedding_softmax(x, embed = False)
    return x, past

  def loadCheckpoint(self, model_path, num_layers,pt=False):
    #if not pt:
    pytorch_model_hash = hashlib.md5(model_path.encode('utf-8')).hexdigest()

    if os.path.exists(pytorch_model_hash):

      print('Loading instead of converting from TensorFlow')

      self.tied_embedding_softmax.load_state_dict(checkpoint['softmax'])
      self.encoder.load_state_dict(checkpoint['encoder'])

      self.tied_embedding_softmax.to('cuda')
      self.encoder.to('cuda')

    else:
      print('Could not find PyTorch checkpoint')
      print('Converting weights and will store the PyTorch checkpoint as ', pytorch_model_hash)


def flipdict(my_map):
    '''
    将输入的字典进行翻转，即将原字典的键和值互换
    '''
    return {v: k for k, v in my_map.items()}



# In[2]:
def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    return logits


def predict_fn(model,inputs,past=None):
    with torch.no_grad():
        inputs = torch.tensor(inputs).cuda()
        output, past = model(inputs, past)
        output = torch.cat([output[:,:,-26:-1],output[:,:,5:6]],2) # remove non-AA token logits
        return output, past


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--code",
        default=None,
        type=int,
        required=True,
        help="control code",
    )
    parser.add_argument(
        "--fname",
        default="samples",
        type=str,
        help="write file (appends family name)",
    )
    parser.add_argument(
        "--num_sample_batches",
        default=None,
        type=int,
        required=True,
        help="nunber of samples",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="nunber of samples",
    )
    parser.add_argument(
        "--gen_length",
        default=None,
        type=int,
        required=True,
        help="generation length",
    )
    parser.add_argument(
        "--top_p",
        default=0.5,
        type=float,
        help="top p for nucleus sampling",
    )
    parser.add_argument(
        "--top_k",
        default=0,
        type=int,
        help="nunber of samples",
    )
    parser.add_argument(
        "--rep_penalty",
        default=1.2,
        type=float,
        help="nunber of samples",
    )
    parser.add_argument('--pt_model', type=str, default='checkpoints/pretrain.pth', help='pretraining model path')
    # parser.add_argument('--generate_version', type=str, default='0', help='generate_version')
    args = parser.parse_args()

    seq_length = 511
    embedding_dim = 1280
    num_layers = 36
    vocab_loc = 'mapping_files/vocab.txt'

    use_py3 = platform.python_version()[0] == '3'
    vocab = open(vocab_loc).readlines() if not use_py3 else open(vocab_loc, encoding='utf-8').read().split('\n')[:-1]
    vocab = list(map(lambda x: x.split(' ')[0], vocab))
    vocab_size = len(vocab)
    print('-----vocab size',vocab_size,'------')

    model = CTRLmodel(vocab_size,embedding_dim)

    print('model initialized')

    ckptnum = '1000000'
    # pt_model = "/export/share/amadani/progen/ckpt/lys_model_sep3_0.pth3"
    pt_model = args.pt_model

    model.load_state_dict(torch.load(pt_model)["model_state_dict"])

    #reader = model.loadCheckpoint(model_path=pt_model, num_layers = num_layers,pt=True)

    print('previous checkpoint loaded')
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters()) #lr, betas

    model.eval()

    with open('mapping_files/aa_to_ctrl_idx.p','rb') as handle:
        aa_to_ctrl_idx = pickle.load(handle)

    ctrl_idx_to_aa = flipdict(aa_to_ctrl_idx)

    kw_lineage = []
    tax_lineage = [args.code]       # 家族标签


    example_seq = [] #seqs[1]
    prefix = [] #seqs[7][:50]
    ref = []#seqs[7][50:100]
    penalty = args.rep_penalty
    topk = args.top_k
    topp = args.top_p

    filename = args.fname + "_code_" + str(args.code) + "_" + str(args.top_p) + ".txt"
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fid = open(filename, 'w')

    for i in range(0,args.num_sample_batches):
        if (i%5)==0:
            print(str(i) + " batches out of " + str(args.num_sample_batches) + " complete")

        seed_seq = [aa_to_ctrl_idx[ii] for ii in prefix]

        generate_num = args.gen_length
        seq_length = min(generate_num, 511)

        text = tax_lineage+kw_lineage+seed_seq
        padded_text = text + [0] * (generate_num - len(text))
        tokens_generated = np.tile(padded_text, (1,1))

        tokens_generated = np.concatenate([tokens_generated]*args.batch_size,0)

        LLs = np.zeros((args.batch_size,generate_num-1))


        past = None
        start_token = 0
        seq_len_so_far = len(text)
        time0 = time.time()
        counter=0
        for token in range(len(text)-1, generate_num-1):


            next_token_logits, past = predict_fn(model, tokens_generated[:, start_token:seq_len_so_far], past)

            next_token_ps = torch.softmax(next_token_logits,-1)

            next_token_logits =next_token_logits[:,-1,:]

            start_token = seq_len_so_far
            seq_len_so_far+=1

            _token = token if token < seq_length else -1


            if penalty>0:
                for id in range(0,next_token_logits.shape[0]):
                    penalized_so_far = set()
                    for _ in range(token-3,token+1):
                        generated_token = tokens_generated[0][_] - (vocab_size-26) # added
                        if generated_token in penalized_so_far:
                            continue
                        if generated_token < 0:
                            continue
                        penalized_so_far.add(generated_token)

                        next_token_logits[id,generated_token] /= penalty



            if topk==1:
                idx = torch.argmax(next_token_logits, dim=-1)
            else:
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=topk,top_p=topp)
                idx = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1).squeeze(1)


            # assign the token for generation
            idx += (vocab_size-26) # added to convert 0 AA to original ctrl idx


            for id in range(next_token_logits.shape[0]):
                tokens_generated[id][token+1] = idx[id].item()


                LLs[id,counter] = torch.log(next_token_ps[id,0,idx[id]-(vocab_size-26)]).item()

            counter+=1

    #    print("generation time: " + str(time.time()-time0))


        #stop token
        ctrl_idx_to_aa[vocab_size-1]="!"
        ctrl_idx_to_aa[5] = "#"



        for id in range(0,len(tokens_generated)):

            tokens_generated_so_far = tokens_generated[id].squeeze()[:token+2]
            tokens_generated_so_far = tokens_generated_so_far[(tokens_generated_so_far>=(vocab_size-26)) & (tokens_generated_so_far<(vocab_size))]

            tokens_generated_so_far = ''.join([ctrl_idx_to_aa[c] for c in tokens_generated_so_far])

            if tokens_generated_so_far.find('!') <0:
                length = len(tokens_generated_so_far)
            else:
                length = tokens_generated_so_far.find('!')

            query = tokens_generated_so_far[:length]


            n_log_p = -1*np.sum(LLs[id,:length+1])/(length+1)


            fid.write(query + "," + str(n_log_p) + "\n")

main()


