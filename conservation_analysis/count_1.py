from Bio import AlignIO
from Bio.Align import AlignInfo
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Bio import SeqIO


def compute_entropy(sequence):
    counts = {}
    for aa in sequence:
        if aa in counts:
            counts[aa] += 1
        else:
            counts[aa] = 1

    entropy = 0
    total = len(sequence)
    for count in counts.values():
        p = count / total
        entropy -= p * np.log2(p)

    return entropy


def compute_conservation(filename):
    # 读取fasta文件并获取所有序列
    sequences = []
    for record in SeqIO.parse(filename, "fasta"):
        sequences.append(str(record.seq))

    # 获取序列长度
    length = len(sequences[0])

    # 计算每个位置的保守性
    conservation = []
    for i in range(length):
        column = [sequence[i] for sequence in sequences]
        entropy = compute_entropy(column)
        normalized_entropy = entropy / np.log2(20)
        conservation.append(1 - normalized_entropy)

    return conservation


# 使用fasta文件名获取保守性分数
entropy_scores = compute_conservation("PF06737_small_align.fas")

# 创建一个位点的序列（1开始）
positions = np.arange(1, len(entropy_scores) + 1)

# 使用 Matplotlib 绘制保守性分数
plt.plot(positions, entropy_scores)
plt.xlabel('Amino Acid Position')
plt.ylabel('Conservation Score')
plt.title('Conservation of Amino Acids in Protein Sequences')
plt.ylim(0, 1)
plt.show()
