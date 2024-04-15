from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
import matplotlib.pyplot as plt
from time import sleep
import pickle
import os

in_path = ["data/samples_p_0.75.txt", "data/samples_p_0.5.txt", "data/samples_p_0.25.txt"]
out_path = ["data/samples_p_0.75.xml", "data/samples_p_0.5.xml", "data/samples_p_0.25.xml"]
plt.rc('font',family='Times New Roman')

# blastp：是使用蛋白质序列与蛋白质数据库中的序列进行比对
# nr: 是NCBI的默认算法，由于原文中没有说明故选择默认
def get_blastp_xml(in_path, out_path, num_seq=100):
    '''调用NCBI的blastp接口，获取xml格式的结果文件

    Params:
        in_path: 原始数据文件
        out_path: 保存结果的文件
        num_seq: 需要处理的序列数
    '''

    origin_txt = open(in_path, "r")
    save_file = open(out_path, "a")

    cnt = 0
    for line in origin_txt.readlines():
        if cnt < num_seq:
            if (cnt + 1) % 5 == 0:
                sleep(60)
            str = line[:line.find(",")]
            result_handle = NCBIWWW.qblast("blastp", "nr", str)
            save_file.write(result_handle.read())
        cnt += 1
        print(cnt)

    save_file.close()
    result_handle.close()


def get_max_identity(path):
    '''解析blastp的xml格式结果文件，获取max_id

    Params:
        path: blastp的xml格式结果文件路径

    Returns:
        result: max_id的分布
        length: 样本数
    '''

    result_handle = open(path, "r")
    blast_records = NCBIXML.parse(result_handle)
    list = []
    for blast_record in blast_records:
        max_id = 0
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                max_id = max(max_id, 100 * hsp.identities // hsp.positives)
        list.append(max_id)
    result_handle.close()

    result = [0 for _ in range(101)]
    for max_id in list:
        result[max_id] += 1
    length = len(list)
    result = [result[i] / len(list) for i in range(101)]
    return result, length


def dump():
    '''将三种概率的结果分别存储到文件中'''

    result25, len25 = get_max_identity("./data/samples_p_0.25.xml")
    result50, len50 = get_max_identity("./data/samples_p_0.5.xml")
    result75, len75 = get_max_identity("./data/samples_p_0.75.xml")
    with open("./data/25.txt", "wb") as f:
        pickle.dump(result25, f)
    with open("./data/50.txt", "wb") as f:
        pickle.dump(result50, f)
    with open("./data/75.txt", "wb") as f:
        pickle.dump(result75, f)
    # return [len25, len50, len75]


def draw_triple():
    '''绘制三种概率的图像，三色图'''

    with open("./data/25.txt", "rb") as f:
        result25 = pickle.load(f)
    with open("./data/50.txt", "rb") as f:
        result50 = pickle.load(f)
    with open("./data/75.txt", "rb") as f:
        result75 = pickle.load(f)

    plt.xlabel("MaxID with any known natural protein", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.bar([i for i in range(40, 101)], result25[40:], alpha=0.7, label="p=0.25", color="grey")
    plt.bar([i for i in range(40, 101)], result50[40:], alpha=0.5, label="p=0.50", color="blue")
    plt.bar([i for i in range(40, 101)], result75[40:], alpha=0.5, label="p=0.75", color="green")
    plt.legend(loc='upper left', prop={'size': 12})
    plt.show()


def draw_single():
    '''绘制三种概率的图像，单色图'''

    with open("./data/25.txt", "rb") as f:
        result25 = pickle.load(f)
    with open("./data/50.txt", "rb") as f:
        result50 = pickle.load(f)
    with open("./data/75.txt", "rb") as f:
        result75 = pickle.load(f)
    len_list = [len(result25), len(result50), len(result75)]
    total_len = sum(len_list)
    result25 = [result25[i] * len_list[0] / total_len for i in range(101)]
    result50 = [result50[i] * len_list[1] / total_len for i in range(101)]
    result75 = [result75[i] * len_list[2] / total_len for i in range(101)]
    result = [result25[i] + result50[i] + result75[i] for i in range(101)]

    plt.xlabel("MaxID with any known natural protein", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    # 刻度值字体大小设置（x轴和y轴同时设置）
    plt.tick_params(labelsize=20)
    plt.bar([i for i in range(40, 101)], result[40:], alpha=1, color="blue")
    plt.tight_layout()
    plt.show()


def draw_respectively():
    '''绘制三种概率的图像，每个p单独一个图像'''
    with open("./data/25.txt", "rb") as f:
        result25 = pickle.load(f)
    with open("./data/50.txt", "rb") as f:
        result50 = pickle.load(f)
    with open("./data/75.txt", "rb") as f:
        result75 = pickle.load(f)

    plt.xlabel("MaxID with any known natural protein", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    # 注释不同的语句，画不同的图像
    # plt.bar([i for i in range(40, 101)], result25[40:], alpha=0.7, label="p=0.25", color="grey")
    # plt.bar([i for i in range(40, 101)], result50[40:], alpha=0.5, label="p=0.50", color="blue")
    plt.bar([i for i in range(40, 101)], result75[40:], alpha=0.5, label="p=0.75", color="green")
    plt.legend(loc='upper left', prop={'size': 20})
    # 刻度值字体大小设置（x轴和y轴同时设置）
    plt.tick_params(labelsize=20)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # dump()

    for i in range(len(out_path)):
        if not os.path.exists(out_path[i]):
            get_blastp_xml(in_path[i], out_path[i])
    # draw_triple()
    # draw_single()
    draw_respectively()
