import os
import torch
import hashlib
import platform
import pickle
import time
import numpy as np
import torch.nn.functional as F
import transformer_lora_gen as pytorch_transformer
from transformer_lora_gen import lora_config
import argparse


use_py3 = platform.python_version()[0] == '3'   # 检查当前运行的Python版本是否为3。如果是，则将变量use_py3设置为True，否则为False


class TiedEmbeddingSoftmax(torch.nn.Module):
    '''
    实现一个绑定的 嵌入-softmax 模型
    '''
    def __init__(self, vocab_size, embedding_size, **kwargs):
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
    def __init__(self, vocab_size, embedding_size, loraconfig):
        super(CTRLmodel,self).__init__()
        # 创建实例
        self.tied_embedding_softmax = TiedEmbeddingSoftmax(vocab_size, embedding_size)
        self.encoder = pytorch_transformer.Encoder(loraconfig=loraconfig)
    
    def forward(self, inputs, past=None):
        # 将输入序列转化为嵌入向量
        x = self.tied_embedding_softmax(inputs, embed=True)
        x, past = self.encoder(x, past)
        x = self.tied_embedding_softmax(x, embed=False)
        return x, past

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

def flipdict(my_map):
    '''
    将输入的字典进行翻转，即将原字典的键和值互换
    '''
    return {v: k for k, v in my_map.items()}

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
        # sorted_logits 中的元素是按照从高到低的顺序排列的 logits 值
        # sorted_indices 中的元素则是对应的排序后的索引，可以用于获取原始 logits 中的元素的排列情况
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
        # remove non-AA token logits，只保留AA-token和stop-token，其中stop-token的ctrl代码为437
        output = torch.cat([output[:,:,-26:-1],output[:,:,437:438]],2) 
        return output, past


# 解析标签，输入为字符串格式，如“2.7.11.1”
# 若输入为“2.7.11”，则扩充为“2.7.11.none”
def parser_labels(label, mapping_files):
    # 将字符串分割成列表
    parts = label.split('.')
    if len(parts) < 4:
        parts += ['none'] * (4 - len(parts))
    # 使用列表推导式创建新列表
    label_list = [part for part in parts]
    result=[]
    for i in label_list:
        if i not in mapping_files:
            raise ValueError(f'Invalid label: {i}')
        result.append(mapping_files[i])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        default=None,
        type=str,
        required=True,
        help="EC number, e.g.2.7.11.2",
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
    parser.add_argument('--lora_model', type=str, default='checkpoints/lora_model.pt', help='lora model path')
    parser.add_argument('--lora_dim', type=int, default= 4, help='lora intrinsic dimension')
    parser.add_argument('--lora_alpha', type=int, default=32, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    args = parser.parse_args()

    embedding_dim = 1280
    num_layers = 36
    # 打开词汇表
    vocab_loc = 'mapping_files/vocab.txt'
    use_py3 = platform.python_version()[0] == '3'
    vocab = open(vocab_loc).readlines() if not use_py3 else open(vocab_loc, encoding='utf-8').read().split('\n')[:-1]
    vocab = list(map(lambda x: x.split(' ')[0], vocab))
    vocab_size = len(vocab)
    print('-----vocab size',vocab_size,'------')
    # lora参数配置
    l_config = lora_config(
        lora_attn_dim=args.lora_dim, 
        lora_attn_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout,
    )
    # 初始化模型结构
    model = CTRLmodel(vocab_size, embedding_dim, l_config)
    print('model initialized')
    # 加载训练好的模型
    pt_model = args.pt_model
    model.load_state_dict(torch.load(pt_model)["model_state_dict"], strict=False)
    print('previous checkpoint loaded')
    lora_model = args.lora_model
    model.load_state_dict(torch.load(lora_model)["model_state_dict"], strict=False)
    print('lora finetune checkpoint loaded')
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters()) #lr, betas
    model.eval()

    with open('mapping_files/aa_to_ctrl_idx.p','rb') as handle:
        aa_to_ctrl_idx = pickle.load(handle)
    with open('mapping_files/ec_level_to_ctrl_v2.p','rb') as handle:
        ec_level_to_ctrl = pickle.load(handle)

    ctrl_idx_to_aa = flipdict(aa_to_ctrl_idx)

    kw_lineage = []
    tax_lineage = parser_labels(args.label, ec_level_to_ctrl)       # EC number

    example_seq = [] #seqs[1]
    prefix = [] #seqs[7][:50]
    ref = []#seqs[7][50:100]
    penalty = args.rep_penalty
    topk = args.top_k
    topp = args.top_p


    filename = args.fname + "_label_" + str(args.label) + "_" + str(args.top_p) + ".txt"
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

        text = tax_lineage+kw_lineage+seed_seq  # 初始序列
        padded_text = text + [0] * (generate_num - len(text))   # 对齐
        tokens_generated = np.tile(padded_text, (1,1))  # 复制padded_text
        tokens_generated = np.concatenate([tokens_generated]*args.batch_size,0) # 在行方向上拼接batch_size 次

        LLs = np.zeros((args.batch_size,generate_num-1))    # 对数似然得分

        past = None
        start_token = 0
        seq_len_so_far = len(text)
        counter=0
        for token in range(len(text)-1, generate_num-1):
            next_token_logits, past = predict_fn(model, tokens_generated[:, start_token:seq_len_so_far], past)
            next_token_ps = torch.softmax(next_token_logits,-1)     # 对最后一维进行softmax，用来求LLscore
            next_token_logits =next_token_logits[:,-1,:]

            start_token = seq_len_so_far
            seq_len_so_far+=1

            if penalty>0:
                for id in range(0,next_token_logits.shape[0]):  # 遍历当前batch中所有元素
                    penalized_so_far = set()    # 存储已经惩罚过的标记
                    for _ in range(token-3,token+1):    # token为当前序列的最后一个位置，即只检查附近4个元素
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

            idx += (vocab_size-26) # added to convert 0 AA to original ctrl idx

            for id in range(next_token_logits.shape[0]):    # 遍历一个batch中所有元素
                tokens_generated[id][token+1] = idx[id].item()
                # TODO：该行之后的代码还没有彻底弄懂
                LLs[id,counter] = torch.log(next_token_ps[id,0,idx[id]-(vocab_size-26)]).item()

            counter+=1

        # stop token
        ctrl_idx_to_aa[vocab_size-1]="!"    # vocab_size-1 = 129406，也就是PAD
        ctrl_idx_to_aa[437] = "#"     # ctrl_code=5，表示STOP标记

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

# # 测试parser_labels函数
# with open('mapping_files/ec_level_to_ctrl_v2.p','rb') as handle:
#     ec_level_to_ctrl = pickle.load(handle)
# label = "2.7"
# result = parser_labels(label, ec_level_to_ctrl)
# print(result)

main()

