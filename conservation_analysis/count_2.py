from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt
import os


def compute_entropy(sequence):
    counts = {}
    total = 0
    for aa in sequence:
        # if aa != '-':
        total += 1
        if aa in counts:
            counts[aa] += 1
        else:
            counts[aa] = 1

    entropy = 0
    for count in counts.values():
        p = count / total
        entropy -= p * np.log2(p)

    return entropy


def compute_conservation(filename):
    # 读取fasta文件并获取所有序列
    sequences = list(SeqIO.parse(filename, "fasta"))

    print(len(sequences))
    # 获取序列长度
    length = len(sequences[0])

    # 计算每个位置的保守性
    conservation_manual = []
    conservation_nature = []
    for i in range(length):
        column_nature = [sequence[i] for sequence in sequences if "|" in sequence.name]
        column_manual = [sequence[i] for sequence in sequences if "|" not in sequence.name]
        if column_manual.count('-') + column_nature.count('-') < (len(column_manual) + len(column_nature)) * 0.2:
            entropy_manual = compute_entropy(column_manual)
            if len(set(column_manual)) == 1:
                normalized_entropy = 0
            else:
                normalized_entropy = entropy_manual / np.log2(len(set(column_manual)))
            conservation_manual.append(normalized_entropy)

            entropy_nature = compute_entropy(column_nature)
            if len(set(column_nature)) == 1:
                normalized_entropy = 0
            else:
                normalized_entropy = entropy_nature / np.log2(len(set(column_nature)))
            conservation_nature.append(normalized_entropy)

    return conservation_manual, conservation_nature


def main(file_name):
    # 使用fasta文件名获取保守性分数
    file_path_nature = rf"D:\programFile\pycharm\progen\lysozyme_dataset\merge\align_data\{file_name}"
    entropy_scores_nature, entropy_scores_manual = compute_conservation(file_path_nature)

    # 创建一个位点的序列（1开始）
    positions_manual = np.arange(1, len(entropy_scores_manual) + 1)
    positions_natrue = np.arange(1, len(entropy_scores_nature) + 1)

    # 使用 Matplotlib 绘制保守性分数
    plt.figure(figsize=(15, 5))
    plt.plot(positions_natrue, entropy_scores_nature, linewidth=3, alpha=0.5)
    plt.plot(positions_manual, entropy_scores_manual, linewidth=3, alpha=0.5)
    plt.legend(['nature', 'manual'])
    plt.xlabel(f'Amino Acid Position({file_name})')
    plt.ylabel('Conservation Score')
    plt.title('Conservation of Amino Acids in Protein Sequences')
    plt.ylim(0, 1)
    plt.savefig(rf"D:\programFile\pycharm\progen\lysozyme_dataset\merge\conservation\{file_name}.png")
    plt.show()


if __name__ == '__main__':
    for file in os.listdir(r"D:\programFile\pycharm\progen\lysozyme_dataset\merge\align_data"):
        if file.endswith(".fas"):
            main(file)
