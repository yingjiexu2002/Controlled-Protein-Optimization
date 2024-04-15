from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import random


def read_npy(seq_path, label_path):
    padded_sequences = []
    labels = []
    padded_sequences = np.load(seq_path)
    labels = np.load(label_path)
    print(seq_path + f"的序列shape为{padded_sequences.shape}, label shape为{labels.shape}")
    return padded_sequences, labels



def plot(x, colors, per, n_iter):
    # 选择颜色调色板
    # https://seaborn.pydata.org/generated/seaborn.color_palette.html
    palette = np.array(sns.color_palette("pastel", 6))
    # pastel, husl, and so on

    # 创建散点图
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, alpha=0.6, c=palette[colors.astype(np.int8)])

    # 添加图例
    legend_labels = ["PF00959", "PF01832", "PF05838", "PF06737", "PF16754", "generation_seq"]
    legend_handles = [plt.Line2D([], [], marker='o', markersize=8, color=c, linestyle='None') for c in palette]
    ax.legend(legend_handles, legend_labels)

    plt.savefig(f'picture/tsne_perplexity{per}_step{n_iter}.png', dpi=120)

    return f, ax


def main():
    print("start: " + str(datetime.now()))

    # 读取人工序列进行T-sne-------------------------------------------------------------------------------------
    pad_PF00959_generation, label_PF00959_generation = read_npy('../dataset/embed/PF00959_generation_sequences.npy', '../dataset/embed/PF00959_generation_labels.npy')
    pad_PF01832_generation, label_PF01832_generation = read_npy('../dataset/embed/PF01832_generation_sequences.npy', '../dataset/embed/PF01832_generation_labels.npy')
    pad_PF05838_generation, label_PF05838_generation = read_npy('../dataset/embed/PF05838_generation_sequences.npy', '../dataset/embed/PF05838_generation_labels.npy')
    pad_PF06737_generation, label_PF06737_generation = read_npy('../dataset/embed/PF06737_generation_sequences.npy', '../dataset/embed/PF06737_generation_labels.npy')
    pad_PF16754_generation, label_PF16754_generation = read_npy('../dataset/embed/PF16754_generation_sequences.npy', '../dataset/embed/PF16754_generation_labels.npy')

    # 对PF01832进行随机取样（因为顺序取的话，画图效果不好）
    random_rows_generation = np.random.choice(pad_PF01832_generation.shape[0], 600, replace=False)
    new_01832_generation = pad_PF01832_generation[random_rows_generation]
    padded_sequences_generation = np.concatenate((pad_PF00959_generation[0:400], new_01832_generation, pad_PF05838_generation[0:140], pad_PF06737_generation[0:300], pad_PF16754_generation[0:120]), axis=0)
    # labels = np.concatenate((label_PF00959, label_PF01832, label_PF05838, label_PF06737, label_PF16754), axis=0)
    labels_generation = np.array([5 for _ in range(len(padded_sequences_generation))])

    # 读取自然序列进行T-sne-------------------------------------------------------------------------------------
    pad_PF00959, label_PF00959 = read_npy('../dataset/embed/PF00959_sequences.npy', '../dataset/embed/PF00959_labels.npy')
    pad_PF01832, label_PF01832 = read_npy('../dataset/embed/PF01832_sequences.npy', '../dataset/embed/PF01832_labels.npy')
    pad_PF05838, label_PF05838 = read_npy('../dataset/embed/PF05838_sequences.npy', '../dataset/embed/PF05838_labels.npy')
    pad_PF06737, label_PF06737 = read_npy('../dataset/embed/PF06737_sequences.npy', '../dataset/embed/PF06737_labels.npy')
    pad_PF16754, label_PF16754 = read_npy('../dataset/embed/PF16754_sequences.npy', '../dataset/embed/PF16754_labels.npy')
    
    random_rows = np.random.choice(pad_PF01832.shape[0], 300, replace=False)
    new_01832 = pad_PF01832[random_rows]
    padded_sequences_natural = np.concatenate((pad_PF00959[0:195], new_01832, pad_PF05838[0:71], pad_PF06737[0:137], pad_PF16754[0:58]), axis=0)
    labels_natural = np.concatenate((label_PF00959[0:195], label_PF01832[0:300], label_PF05838[0:71], label_PF06737[0:137], label_PF16754[0:58]), axis=0)

    # 文件读取结束----------------------------------------------------------------------------------------------

    # 整个输入序列，人工序列在前，自然序列在后
    padded_sequences = np.concatenate((padded_sequences_generation, padded_sequences_natural), axis=0)
    labels = np.concatenate((labels_generation, labels_natural), axis=0)

    # 输出文件样本
    for i, pad_seq in enumerate(padded_sequences):
        if i > 0:
            break
        print(f"进行t-sne的单个序列样式为：{pad_seq}")
        print(f"进行t-sne的单个序列长度为：{len(pad_seq)}")
    print("参与降维分析的样本数量形式" + str(padded_sequences.shape))
    print("参与t-sne的labels的shape："+str(labels.shape))


    for n_iter in [2000, 5000]:
        for per in [10, 30, 50, 70, 100]:
            print("开始t-sne：" + str(datetime.now()))
            # 使用t-SNE进行降维
            tsne = TSNE(n_components=2, perplexity=per, n_iter=n_iter, init='pca')
            # # 将填充后的序列转换为NumPy数组
            # padded_sequences = np.array([eval(seq) for seq in padded_sequences], dtype=np.float32)
            X_tsne = tsne.fit_transform(padded_sequences)

            print("开始绘制图像：" + str(datetime.now()))
            plot(X_tsne, labels, per, n_iter)
            # plt.show()


main()