import datetime
import os
import random
import time

import torch
import numpy as np
from Bio import SeqIO
from tape import TAPETokenizer
from tape.models.modeling_bert import ProteinBertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

# 模型存储到的路径
output_dir = 'model_save/'
# 目录不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
max_seq_length = 256
batch_size = 32
learning_rate = 5e-5
adam_epsilon = 1e-8
epochs = 4
seed_val = 42

pfam = ['PF00959', 'PF01832', 'PF05838', 'PF06737', 'PF16754']
pfam_id = [0, 1, 2, 3, 4]


def proc_seq(result, seq):
    token_ids = torch.tensor(np.array([tokenizer.encode(seq)]))
    output = new_model(token_ids)
    sequence_output = output[0]
    result.append(sequence_output[0][0].detach().numpy())


# def get_embed():
#     for fam, id in zip(pfam, pfam_id):
#         result = []
#         # with open("code/protein_embed/test.fasta", "r") as generated:
#         with open(f"code/protein_embed/generate_fasta/{fam}_fasta/{fam}_combined.fasta", "r") as generated:
#             seq = ""
#             for line in generated.readlines():
#                 if line[0] == ">" and seq != "":
#                     proc_seq(result, seq)
#                     seq = ""
#                     if len(result) % 100 == 0:
#                         print(len(result))
#                 elif line[0] == ">":
#                     continue
#                 else:
#                     seq += line[:-1]
#             proc_seq(result, seq)

#         embeds = np.array(result)
#         labels = np.array([id for _ in range(len(result))])
#         np.save(f"code/tape-bert/embed/{fam}_generated_sequences.npy", embeds)
#         np.save(f"code/tape-bert/embed/{fam}_generated_labels.npy", labels)
#         print(embeds.shape)
#         print(labels.shape)


def get_embed():
    new_model = ProteinBertForSequenceClassification.from_pretrained(output_dir)
    print('模型已加载完成')
    new_model.to(device)
    for fam, id in zip(pfam, pfam_id):
        result = []
        # with open(f"../lysozyme_dataset/{fam}.fasta", "r") as generated:
        print("running:"+str(fam))
        data_set_PF = get_dataset(f"../protein_embed/lysozyme_dataset/{fam}.fasta", id)
        validation_dataloader = DataLoader(
            data_set_PF,  # 验证样本
            sampler=SequentialSampler(data_set_PF),  # 顺序选取小批量
            batch_size=1
        )
        for step, batch in enumerate(validation_dataloader):
            if step % 200 == 0:
                print('step='+str(step))
            # 准备输入数据，并将其拷贝到 gpu 中
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            inputs = {"input_ids": b_input_ids, "input_mask": b_input_mask, "targets": b_labels}
            with torch.no_grad():
                outputs , seq = new_model(**inputs)

            seq_c = seq.to('cpu')
            result.append(seq_c[0][0].detach().numpy())
            print(seq)
            exit()
            # result.append(seq_c[1][0].detach().numpy())
            # result.append(seq_c[2][0].detach().numpy())
            # result.append(seq_c[3][0].detach().numpy())

        embeds = np.array(result)
        labels = np.array([id for _ in range(len(result))])
        np.save(f"embed/{fam}_sequences.npy", embeds)
        np.save(f"embed/{fam}_labels.npy", labels)
        print(embeds.shape)
        print(labels.shape)


# def read_seq_from_npy(npy_path):
#     sequences = []
#     with open(npy_path) as generated:
#         seq = ""
#         for line in generated.readlines():
#             if line[0] == ">" and seq != "":
#                 proc_seq(sequences, seq)
#                 seq = ""
#                 if len(sequences) % 100 == 0:
#                     print(len(sequences))
#             elif line[0] == ">":
#                 continue
#             else:
#                 seq += line[:-1]
#         proc_seq(sequences, seq)
#     return np.array(sequences)


def get_dataset(fasta_path, label):
    # 将数据集分完词后存储到列表中
    all_input_ids = []
    attention_masks = []
    sentences, labels = read_seq_and_label_from_fasta(fasta_path, label)

    for sent in sentences:
        input_ids = tokenizer.encode(sent).tolist()
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[0:max_seq_length]
        # input_ids = [tokenizer.vocab['<cls>']] + input_ids + [tokenizer.vocab['<sep>']]

        # 获取mask
        pad_len = max_seq_length - len(input_ids)
        mask = torch.tensor([1] * max_seq_length)
        if pad_len > 0:
            input_ids = input_ids + [tokenizer.vocab['<pad>']] * pad_len
            mask[-pad_len:] = 0

        # 将编码后的文本加入到列表
        all_input_ids.append(input_ids)

        # 将文本的 attention mask 也加入到 attention_masks 列表
        attention_masks.append(mask.tolist())

    # 将列表转换为 tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    labels = torch.tensor(labels)

    # # 输出第 1 行文本的原始和编码后的信息
    # print('Original: ', sentences[0])
    # print('Token IDs:', all_input_ids[0])
    dataset = TensorDataset(all_input_ids, attention_masks, labels)
    return dataset


def read_label_from_npy(label_path):
    return np.load(label_path)


def read_seq_and_label_from_fasta(filename, label):
    # 读取fasta文件并编码序列
    sequences = []
    labels = []
    with open(filename, 'r') as file:
        for record in SeqIO.parse(file, 'fasta'):
            sequence = str(record.seq)
            sequences.append(sequence)
            labels.append(label)
    return sequences, labels


# 根据预测结果和标签数据来计算准确率
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))

    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(model, train_dataloader, validation_dataloader, optimizer, scheduler):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # 存储训练和评估的 loss、准确率、训练时长等统计指标,
    training_stats = []
    # 统计整个训练时长
    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # 统计单次 epoch 的训练时间
        t0 = time.time()

        # 重置每次 epoch 的训练总 loss
        total_train_loss = 0

        # 将模型设置为训练模式。这里并不是调用训练接口的意思
        # dropout、batchnorm 层在训练和测试模式下的表现是不同的 (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # 训练集小批量迭代
        for step, batch in enumerate(train_dataloader):

            # 每经过40次迭代，就输出进度信息
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # 准备输入数据，并将其拷贝到 gpu 中
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
            model.zero_grad()

            # 前向传播
            # 文档参见:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # 该函数会根据不同的参数，会返回不同的值。 本例中, 会返回 loss 和 logits -- 模型的预测结果
            # 为模型的输入创建一个字典
            inputs = {"input_ids": b_input_ids, "input_mask": b_input_mask, "targets": b_labels}
            outputs = model(**inputs)

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            loss = loss[0]
            # 累加 loss
            total_train_loss += loss.item()
            # 反向传播
            loss.backward()
            # 梯度裁剪，避免出现梯度爆炸情况
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 更新参数
            optimizer.step()
            # 更新学习率
            scheduler.step()

        # 平均训练误差
        avg_train_loss = total_train_loss / len(train_dataloader)

        # 单次 epoch 的训练时长
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # 完成一次 epoch 训练后，就对该模型的性能进行验证

        print("")
        print("Running Validation...")

        t0 = time.time()

        # 设置模型为评估模式
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        step = 0
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # 将输入数据加载到 gpu 中
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # 为模型的输入创建一个字典
            inputs = {"input_ids": b_input_ids, "input_mask": b_input_mask, "targets": b_labels}

            # 评估的时候不需要更新参数、计算梯度
            with torch.no_grad():
                outputs = model(**inputs)

            loss = outputs[0]
            loss = loss[0]
            logits = outputs[1]
            # 累加 loss
            total_eval_loss += loss.item()

            # 将预测结果和 labels 加载到 cpu 中计算
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # 计算准确率
            total_eval_accuracy += flat_accuracy(logits, label_ids)

            if (step % 100) == 0:
                print('step='+str(step))
                print("total_eval_accuracy="+str(total_eval_accuracy))
                print("flat_accuracy="+str(flat_accuracy(logits, label_ids)))
            step += 1

        # 打印本次 epoch 的准确率
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  total_eval_accuracy="+str(total_eval_accuracy))
        print('  len(validation_dataloader)='+str(len(validation_dataloader)))
        print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

        # 统计本次 epoch 的 loss
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        # 统计本次评估的时长
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.4f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # 记录本次 epoch 的所有统计信息
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
    # 保存模型
    model_to_save = model.module if hasattr(model, 'module') else model  # 考虑到分布式/并行（distributed/parallel）训练
    model_to_save.save_pretrained(output_dir)


def test(model, prediction_dataloader):
    # 预测测试集
    # 依然是评估模式
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []
    print("开始预测")

    total_eval_accuracy = 0
    # 预测
    for batch in prediction_dataloader:
        # 将数据加载到 gpu 中
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # 为模型的输入创建一个字典
        inputs = {"input_ids": b_input_ids, "input_mask": b_input_mask, "targets": b_labels}

        # 不需要计算梯度
        with torch.no_grad():
            # 前向传播，获取预测结果
            outputs = model(**inputs)

        logits = outputs[1]

        # 将结果加载到 cpu 中
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # 计算准确率
        total_eval_accuracy += flat_accuracy(logits, label_ids)

        # 存储预测结果和 labels
        predictions.append(logits)
        true_labels.append(label_ids)
    # 打印本次 epoch 的准确率
    avg_val_accuracy = total_eval_accuracy / len(prediction_dataloader)
    print("  Accuracy: {0:.4f}".format(avg_val_accuracy))
    print('    DONE.')


def plot():
    # 三个类别的名称和对应的正确率
    categories = ['Class A', 'Class B', 'Class C']
    accuracies = [85, 92, 78]  # 这里的数值是示例，请根据实际情况修改

    # 创建一个圆盘图
    plt.figure(figsize=(6, 6))  # 设置图的大小

    # 绘制圆盘图
    plt.pie(accuracies, labels=categories, autopct='%1.1f%%', startangle=140, colors=['blue', 'green', 'orange'])

    # 添加标题
    plt.title('Accuracy by Category')

    # 显示图
    plt.axis('equal')  # 保证图是一个正圆
    plt.show()


def main():
    # # 为训练和验证集创建 Dataloader，对训练样本随机洗牌
    # # 训练集
    # PF00959_dataset = get_dataset('../protein_embed/lysozyme_dataset/PF00959.fasta', 0)
    # PF01832_dataset = get_dataset('../protein_embed/lysozyme_dataset/PF01832.fasta', 1)
    # PF05838_dataset = get_dataset('../protein_embed/lysozyme_dataset/PF05838.fasta', 2)
    # PF06737_dataset = get_dataset('../protein_embed/lysozyme_dataset/PF06737.fasta', 3)
    # PF16754_dataset = get_dataset('../protein_embed/lysozyme_dataset/PF16754.fasta', 4)
    # train_dataset = ConcatDataset([PF00959_dataset, PF01832_dataset, PF05838_dataset, PF06737_dataset, PF16754_dataset])
    # print('训练集已加载')

    # # 验证集
    # PF06737_dataset_val = get_dataset('../protein_embed/generate_fasta/PF06737_fasta/PF06737_combined.fasta', 3)
    # PF16754_dataset_val = get_dataset('../protein_embed/generate_fasta/PF16754_fasta/PF16754_combined.fasta', 4)
    # val_dataset = ConcatDataset([PF06737_dataset_val, PF16754_dataset_val])
    # print("验证集已加载")

    # train_dataloader = DataLoader(
    #     train_dataset,  # 训练样本
    #     sampler=RandomSampler(train_dataset),  # 随机小批量
    #     batch_size=batch_size  # 以小批量进行训练
    # )

    # # 验证集不需要随机化，这里顺序读取就好
    # validation_dataloader = DataLoader(
    #     val_dataset,  # 验证样本
    #     sampler=SequentialSampler(val_dataset),  # 顺序选取小批量
    #     batch_size=batch_size
    # )


    # # 加载模型
    # model = ProteinBertForSequenceClassification.from_pretrained(
    #     "bert-base",
    #     num_labels=5,  # 分类数
    #     output_attentions=False,  # 模型是否返回 attentions weights.
    #     output_hidden_states=False,  # 模型是否返回所有隐层状态.
    # )
    # print('模型已初始化')
    # # 在 gpu 中运行该模型
    # model.cuda()

    # # 优化器
    # optimizer = AdamW(model.parameters(),
    #                   lr=learning_rate,  # args.learning_rate - default is 5e-5
    #                   eps=adam_epsilon  # args.adam_epsilon  - default is 1e-8
    #                   )

    # # 总的训练样本数
    # total_steps = len(train_dataloader) * epochs

    # # 创建学习率调度器
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=0,
    #                                             num_training_steps=total_steps)

    # train(model, train_dataloader, validation_dataloader, optimizer, scheduler)


    # 测试集
    PF00959_dataset_test = get_dataset('../protein_embed/generate_fasta/PF00959_fasta/PF00959_combined.fasta', 0)
    PF01832_dataset_test = get_dataset('../protein_embed/generate_fasta/PF01832_fasta/PF01832_combined.fasta', 1)
    PF05838_dataset_test = get_dataset('../protein_embed/generate_fasta/PF05838_fasta/PF05838_combined.fasta', 2)
    PF06737_dataset_test = get_dataset('../protein_embed/generate_fasta/PF06737_fasta/PF06737_combined.fasta', 3)
    PF16754_dataset_test = get_dataset('../protein_embed/generate_fasta/PF16754_fasta/PF16754_combined.fasta', 4)
    test_dataset = ConcatDataset(
        [PF00959_dataset_test, PF01832_dataset_test, PF05838_dataset_test, PF06737_dataset_test, PF16754_dataset_test])
    print('测试集已加载')

    # 测试集
    test_dataloader = DataLoader(
        test_dataset,  # 验证样本
        sampler=SequentialSampler(test_dataset),  # 顺序选取小批量
        batch_size=batch_size
    )

    new_model = ProteinBertForSequenceClassification.from_pretrained(output_dir)
    print('模型已加载完成')
    new_model.to(device)
    test(new_model, test_dataloader)


# main()
get_embed()