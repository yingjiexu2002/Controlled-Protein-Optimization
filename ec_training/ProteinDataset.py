import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

class ProteinDataset(Dataset):

    def __init__(self, pklpath, firstAAidx, transformFull=None, transformPartial=None, transformNone=None, evalTransform=None):
        with open(pklpath, 'rb') as handle:     # pklpath：包含蛋白质数据的pickle文件的路径
            self.data_chunk = pickle.load(handle)
        self.uids = list(self.data_chunk.keys())    # 将蛋白质数据块唯一标识符转化为list
        self.transformFull = transformFull  # 完整转换，用于处理整个序列
        self.transformPartial = transformPartial    # 部分转换，用于处理部分序列
        self.transformNone = transformNone  # 无转换，不进行任何处理
        self.evalTransform = evalTransform  # 评估转换，用于在评估模式下使用
        self.firstAAidx = firstAAidx        # 起始氨基酸的索引
        
        self.trainmode=False    # 训练模式
        if self.evalTransform==None: self.trainmode=True

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        if self.trainmode:  # 默认为True
            # # 随机选择转换函数
            # randnum = np.random.random()    # 生成一个0-1之间的随机数
            # transformObj = self.transformNone   # 设置转换对象，为transformProtein类
            # # 根据随机数的值，决定使用哪个转换对象
            # if randnum>0.25:
            #     transformObj = self.transformFull
            # elif randnum>0.1:
            #     transformObj = self.transformPartial

            # 微调阶段，直接全部转换
            transformObj = self.transformFull   
        # else:
        #     transformObj = self.evalTransform

        sample_arr, existence, padIndex = transformObj.transformSample( self.data_chunk[self.uids[idx]] )   # 返回样本数组、存在性标识、填充索引
        sample_arr = np.array(sample_arr)
        inputs = sample_arr[:-1]    # 取[0, n]的[0, n-1]部分，表示样本数据    
        outputs = sample_arr[1:]    # 取label部分

        if existence in set({0,1}):
            existence = 2
        else:
            existence = 1
        begAAindex = np.argwhere(inputs>=self.firstAAidx)[0][0] # 找到inputs中第一个大于或等于self.firstAAidx的元素的索引，并将其赋值给begAAindex
        
        inputs = torch.from_numpy(inputs)
        outputs = torch.from_numpy(outputs)
        return inputs, outputs, existence, padIndex, begAAindex

# 查看uid的内容
# with open('dataset/train_data.p', 'rb') as handle:     # pklpath：包含蛋白质数据的pickle文件的路径
#     data_chunk = pickle.load(handle)
# uids = list(data_chunk.keys())    # 将蛋白质数据块唯一标识符转化为list
# file = open("output.txt", "w")
# file.write(str(uids))
# file.close