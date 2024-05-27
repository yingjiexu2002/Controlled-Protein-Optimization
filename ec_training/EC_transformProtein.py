import pickle
import os
import numpy as np
import random

class transformProtein:
    """
    mapfold:映射文件夹路径
    maxSampleLength:最大样本长度
    selectSwiss:选择 Swiss 数据库的比例
    selectTrembl:选择 Trembl 数据库的比例
    """

    def __init__(self, mapfold = 'mapping_files/', maxSampleLength = 512, 
                 dropRate = 0.0, noflipseq = False):
        #类内局部变量的初始化
        self.maxSampleLength = maxSampleLength
        self.dropRate = dropRate
        self.noflipseq = noflipseq

        # 打开氨基酸映射pickle文件
        with open(os.path.join(mapfold,'aa_to_ctrl_idx.p'),'rb') as handle:
            self.aa_to_ctrl = pickle.load(handle)
        # 打开标签映射pickle文件
        with open(os.path.join(mapfold,'ec_level_to_ctrl.p'),'rb') as handle:
            self.ec_level_to_ctrl = pickle.load(handle)

    def transformSeq(self, seq):
        """
        Transform the amino acid sequence. Currently only reverses seq--eventually include substitutions/dropout
        """
        if self.noflipseq:
            return seq
        if np.random.random()>=0.5:
            # 以50%的概率对序列进行翻转
            seq = seq[::-1]
        return seq


    # 用于转换和增强样本数据
    def transformSample(self, proteinDict, justidx = True):
        """
        微调时使用的样本编码函数，在原来的基础上删除了对Swiss和trembl数据库标签的处理，仅仅处理家族标签
        将存在级别设置为3。
        用PAD来填充。
        返回编码样本（分类标签、关键词、序列）和存在级别。”
        """
        existence = 3       # 表示样本的存在级别

        # kw标签为空
        kws = {}
        # taxa标签为EC编号
        taxa = proteinDict['EC_number']
        # 以一定的概率翻转序列
        seq = self.transformSeq(proteinDict['seq'])

        code_PAD = self.ec_level_to_ctrl['PAD']
        code_stop = self.ec_level_to_ctrl['STOP']

        encodedSample = []
        
        if justidx: # 默认为True
            # 将所有标签编码
            for tax_line in taxa:
                encodedSample.append(self.ec_level_to_ctrl[tax_line])
            seq_idx = 0
            # 将氨基酸序列进行编码
            while (len(encodedSample)<self.maxSampleLength) and (seq_idx<len(seq)):
                encodedSample.append(self.aa_to_ctrl[seq[seq_idx]])
                seq_idx += 1
            if len(encodedSample)<self.maxSampleLength: # 加入序列结束标记，用于防止生成停不下来的问题
                encodedSample.append(code_stop)
            thePadIndex = len(encodedSample)
            while len(encodedSample)<self.maxSampleLength: # add PAD (index is length of vocab)
                encodedSample.append(code_PAD)

        return encodedSample, existence, thePadIndex

# # 测试函数功能是否正常
# with open('train_data/train_data_ec_label.p', 'rb') as handle:     # pklpath：包含蛋白质数据的pickle文件的路径
#     train_chunk = pickle.load(handle)
# uid = 0
# obj = transformProtein(dropRate = 0.0)
# print(train_chunk[uid])
# print(obj.transformSample(train_chunk[uid]))