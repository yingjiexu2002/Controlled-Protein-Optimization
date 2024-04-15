import os.path

from Bio import SeqIO
import random


def random_select_sequences(input_file, output_file, num_sequences):
    # 读取fasta文件并获取所有序列
    sequences = list(SeqIO.parse(input_file, "fasta"))

    # 随机选择指定数量的序列
    selected_sequences = random.sample(sequences, num_sequences)

    # 将选定的序列写入新的fasta文件
    with open(output_file, "w") as f:
        SeqIO.write(selected_sequences, f, "fasta")


# 输入文件路径、输出文件路径和所需选择的序列数量
property = 'manual'
input_file = rf"D:\programFile\pycharm\progen\lysozyme_dataset\{property}\clean_data"
output_file = rf"D:\programFile\pycharm\progen\lysozyme_dataset\{property}\small_clean_data"
file_name = 'PF06737.fasta'
num_sequences = 500


# 调用函数生成新的fasta文件
random_select_sequences(os.path.join(input_file, file_name), os.path.join(output_file, file_name), num_sequences)
