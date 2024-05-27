import os
import random

from Bio import SeqIO


def get_sequences_from_fasta(fasta_path):
    sequences = []
    with open(fasta_path, 'r') as file:
        for record in SeqIO.parse(file, 'fasta'):
            sequence = str(record.seq)
            sequences.append(sequence)
    return sequences


def get_sequences_from_txt(file_path):
    sequences = []

    with open(file_path, 'r') as txt:
        for line in txt:
            # 使用逗号分割每一行，提取序列部分
            parts = line.strip().split(',')
            if len(parts) >= 1:
                sequence = parts[0]
                sequences.append(sequence)
    return sequences


def write_sequences_to_file(file_path, sequences1, sequences2, label1=0, label2=1):
    # 写入到txt文件中并编码
    with open(file_path, 'w') as txt:
        for sequence in sequences1:
            txt.write(f'{sequence},{label1}\n')
        for sequence in sequences2:
            txt.write(f'{sequence},{label2}\n')
    print("写入成功！")


def combine_txt():
    file_list = [r"../generation_dataset/txt_version/combined_samples_code_0.txt",
                 r"../generation_dataset/txt_version/combined_samples_code_1.txt",
                 r"../generation_dataset/txt_version/combined_samples_code_2.txt",
                 r"../generation_dataset/txt_version/combined_samples_code_3.txt",
                 r"../generation_dataset/txt_version/combined_samples_code_4.txt"]  # 替换为你的文件路径列表
    output_file = r'../generation_dataset/txt_version/all_family_combined_samples.txt'  # 替换为合并后的文件名
    with open(output_file, 'w') as output_txt:
        for file_path in file_list:
            with open(file_path, 'r') as input_txt:
                output_txt.write(input_txt.read())
    print("Files merged successfully.")


def split_txt_file(input_file, output_dir, train_ratio=0.8):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 打乱顺序
    random.shuffle(lines)

    # 划分训练集和验证集
    train_size = int(len(lines) * train_ratio)
    train_lines = lines[:train_size]
    val_lines = lines[train_size:]

    # 写入训练集文件
    train_file = os.path.join(output_dir, 'train.txt')
    with open(train_file, 'w') as f:
        f.writelines(train_lines)

    # 写入验证集文件
    val_file = os.path.join(output_dir, 'dev.txt')
    with open(val_file, 'w') as f:
        f.writelines(val_lines)

    print(f"Train set saved to {train_file}")
    print(f"Validation set saved to {val_file}")


def main():
    # 示例用法
    fasta_file1 = 'train_test/sample_merged_file.txt'  # 第一个文件名
    fasta_file2 = r"..\lysozyme_dataset\combined.fasta"  # 第二个文件名
    output_file = r'train_test\all_sequences_and_labels.txt'  # 输出文本文件名

    # 第一个序列为人工序列
    sequences1 = get_sequences_from_txt(fasta_file1)
    # 第二个序列为自然序列
    sequences2 = get_sequences_from_fasta(fasta_file2)

    write_sequences_to_file(output_file, sequences1, sequences2, label1=0, label2=1)


# main()
# 调用函数进行划分
# input_file = 'train_test/all_sequences_and_labels.txt'
# output_dir = 'train_test'
# train_ratio = 0.8  # 指定训练集比例
# split_txt_file(input_file, output_dir, train_ratio)
combine_txt()