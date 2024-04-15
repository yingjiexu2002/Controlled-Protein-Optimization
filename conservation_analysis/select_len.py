from Bio import SeqIO
import os
import matplotlib.pyplot as plt
from collections import Counter
import random


def draw(sequences, fileName):
    # 计算每个序列的长度
    lengths = [len(seq) for seq in sequences]

    # 使用Counter统计每个长度的序列数量
    length_counts = Counter(lengths)
    # 获取长度和对应的数量
    lengths = list(length_counts.keys())
    counts = list(length_counts.values())
    # 创建一个新的图表
    plt.figure()
    # 创建柱状图
    plt.bar(lengths, counts)
    # 添加标题和标签
    plt.title(f'Sequence Lengths {fileName}')
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.xlim(0, 1000)
    plt.ylim(0, 200)
    # 显示图表
    plt.show()


def random_select_sequences(property, file_name, num_sequences):
    input_file_path = rf"D:\programFile\pycharm\progen\lysozyme_dataset\{property}\clean_data\{file_name}"
    output_file_path = rf"D:\programFile\pycharm\progen\lysozyme_dataset\{property}\small_clean_data\{file_name}"
    # 读取fasta文件并获取所有序列
    sequences = list(SeqIO.parse(input_file_path, "fasta"))

    # 随机选择指定数量的序列
    if num_sequences > len(sequences):
        num_sequences = len(sequences)
    selected_sequences = random.sample(sequences, num_sequences)

    # 将选定的序列写入新的fasta文件
    with open(output_file_path, "w") as f:
        SeqIO.write(selected_sequences, f, "fasta")

    print(f"random select {file_name} sequences done!")


def main(property, file_name):
    print("processing: ", file_name, "...")
    # 定义长度范围
    min_length = 60  # 最小长度
    max_length = 256  # 最大长度

    # 读取fasta文件
    origin_sequences = list(SeqIO.parse(os.path.join(rf'D:\programFile\pycharm\progen\lysozyme_dataset\{property}\data', file_name), 'fasta'))
    print("原始序列数: ", len(list(origin_sequences)))
    draw(origin_sequences, file_name)

    # 过滤出特定长度范围内的序列
    filtered_sequences = [seq for seq in list(origin_sequences) if min_length <= len(seq) <= max_length]
    print("过滤后序列数: ", len(filtered_sequences))
    draw(filtered_sequences, file_name)
    # 保存到新的fasta文件
    SeqIO.write(filtered_sequences,
                os.path.join(rf'D:\programFile\pycharm\progen\lysozyme_dataset\{property}\clean_data', file_name), 'fasta')


if __name__ == '__main__':
    my_property_nature = 'nature'
    my_property_manual = 'manual'
    my_num_sequences = 1000
    for my_property in [my_property_nature, my_property_manual]:
        print("processing: ", my_property, "...")
        for my_file_name in os.listdir(rf'D:\programFile\pycharm\progen\lysozyme_dataset\{my_property}\data'):
            main(my_property, my_file_name)
            random_select_sequences(my_property, my_file_name, my_num_sequences)
            print("---------------------------------------------------")
        print("===================================================")


