import argparse
import glob
import json
import logging
import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from tape import ProteinBertForValuePrediction, TAPETokenizer, ProteinBertConfig, ProteinBertModel, \
    ProteinBertForSequenceClassification

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors


# 计算分类模型的准确率（accuracy）和 F1 分数
# 两个参数表示模型的预测结果和真实标签
def acc_and_f1(preds, labels):
    # 确保输入的预测结果和标签的数量是相同的
    assert len(preds) == len(labels)
    acc = simple_accuracy(preds, labels)
    # f1_score 是用于评估分类模型性能的一种常用指标，用于衡量模型的准确率和召回率的平衡情况
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


from sklearn.metrics import matthews_corrcoef, f1_score


# 计算预测正确的比例
def simple_accuracy(preds, labels):
    return (preds == labels).mean()


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
        BertConfig,
        XLNetConfig,
        XLMConfig,
        RobertaConfig,
        DistilBertConfig,
        AlbertConfig,
        XLMRobertaConfig,
        FlaubertConfig,
    )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, "", BertTokenizer),
}

# 设置随机数种子
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    tb_writer = SummaryWriter()

    # 计算训练的批大小
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # 创建相应的数据采样器
    train_sampler = RandomSampler(train_dataset)
    # 创建用于训练的数据加载器
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)


    # 根据设置的最大训练步数或者最大训练周期数计算总的训练步数
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    
    # Prepare optimizer and schedule (linear warmup and decay)
    # 定义不需要权重衰减的参数
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    # 创建一个 AdamW 优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # 创建一个学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    # 检查之前是否已保存优化器和调度器的状态
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # 检查是否存在之前的检查点，如果存在则从检查点继续训练。
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        # # 设置 global_step 为从模型路径中最后一个保存的检查点的 global_step
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    val_acc_list = []
    # 创建文件保存评估的准确率
    df = pd.DataFrame(columns=['step','evaluating accuracy'])#列名
    df.to_csv(os.path.join(args.output_dir, "eval_acc.csv"),index=False) #路径可以根据需要更改
    
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # 创建一个用于循环遍历训练周期的迭代器
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = train_dataloader
        for step, batch in enumerate(epoch_iterator):
             # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            # 设置为训练模式
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # 为模型的输入创建一个字典
            inputs = {"input_ids": batch[0], "input_mask": batch[1], "targets": batch[2]}

            # 调用模型进行前向计算
            outputs = model(**inputs)

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            if not args.regression:
                loss = loss[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            # 计算梯度
            loss.backward()
            tr_loss += loss.item()
            # 如果达到梯度累积的步数
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # 对梯度进行裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # 日志记录
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (args.local_rank == -1 and args.evaluate_during_training and global_step % (args.logging_steps * 20) == 0):  # Only evaluate when single GPU otherwise metrics may not average well
                        _, results = evaluate(args, model, tokenizer)
                        val_acc_list.append(results['acc'])
                        print('max so far - ' + str(np.max(val_acc_list)))
                        # 保存当前步数和准确率
                        tmp_list = [global_step, results['acc']]
                        data = pd.DataFrame([tmp_list])
                        data.to_csv(os.path.join(args.output_dir, "eval_acc.csv"),mode='a',header=False,index=False)
                        # for key, value in results.items():
                        #     eval_key = "eval_{}".format(key)
                        #     logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    # for key, value in logs.items():
                    #     tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))


                # 保存模型
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    print("Save model checkpoint")
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    # tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    # 返回训练的全局步数和平均损失
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, test=False, prefix="", eval_on_train=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    emnlp_analysis = {'class_0_logits': [], 'class_1_logits': [], 'labels': []}

    results = {}
    scores = []
    seqs = []

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        # 循环遍历评估任务和对应的输出目录
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, test=test, evaluate=(not eval_on_train))

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in eval_dataloader:
            model.eval()

            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "input_mask": batch[1], "targets": batch[2]}
                outputs = model(**inputs)

                if args.regression:
                    tmp_eval_loss, logits = outputs[:2]
                else:
                    tmp_eval_loss = outputs[0][0]
                    logits = outputs[1]
                    scores += torch.softmax(logits, -1)[:, 1].cpu().numpy().tolist()

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["targets"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["targets"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        if args.regression:
            preds = np.squeeze(preds)
            out_label_ids = np.squeeze(out_label_ids)
            loss = np.mean((preds - out_label_ids) ** 2)
            result = {"MSE": loss}
            # import pdb; pdb.set_trace()

        else:
            preds = np.argmax(preds, axis=1)
            try:

                result = compute_metrics(eval_task, preds, out_label_ids)
            except:
                result = compute_metrics('sst-2', preds, out_label_ids)
        results.update(result)
        if eval_task == 'cola':
            result.update({'acc': simple_accuracy(preds, out_label_ids)})
        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")

        if not args.regression:
            import sklearn

            print('***** Confusion Matrix *****')
            print(sklearn.metrics.confusion_matrix(out_label_ids, preds))

            result['conf_matrix'] = sklearn.metrics.confusion_matrix(out_label_ids, preds)

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        if True:  # args.custom_mode:
            result['eval_loss'] = eval_loss
            # experiments for HPO
            output_eval_file = os.path.join(args.data_dir, "ext_model_eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

    return scores, result


def load_and_cache_examples(args, task, tokenizer, test=False, evaluate=False):
    '''
    输入参数的意义如下：
    test：是否使用测试集
    evaluate：是否执行评价任务。
    '''
    try:
        processor = processors[task]()
        output_mode = output_modes[task]
    except:
        processor = processors['sst-2']()
        output_mode = output_modes['sst-2']

    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
    if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
        # HACK(label indices are swapped in RoBERTa pretrained model)
        label_list[1], label_list[2] = label_list[2], label_list[1]

    # 此时执行训练任务，读取的文件为data_dir路径下的train.tsv文件
    if evaluate == False:
        logger.info("  dealing dataset: %s", "train")
        convert_file_parameter(args.synth_file, "train")
        examples = processor.get_train_examples(args.data_dir)
    # 执行评价任务，当test为true时,使用测试集
    if evaluate and test:
        logger.info("  dealing dataset: %s", "test")
        # 将测试集转化为需要的格式，并存放到tmp/test文件夹
        convert_file_parameter(args.synth_file, "test/dev", test=True)
        examples = processor.get_dev_examples(args.data_dir + "/test")
    # 执行评价任务，当test为false时,使用验证集
    if evaluate and not test:
        logger.info("  dealing dataset: %s", "dev")
        convert_file_parameter("train_test/dev.txt", "dev", test=False)
        examples = processor.get_dev_examples(args.data_dir)


    all_mask = []
    all_input_ids = []
    all_labels = []

    # 处理读取的数据
    for i in range(0, len(examples)):
        input_ids = tokenizer.encode(examples[i].text_a).tolist()

        if len(input_ids) > (args.max_seq_length):
            input_ids = input_ids[0:1] + input_ids[1:args.max_seq_length - 1] + input_ids[-1:]
            # print("Warning, truncating sequence!!!!!")

        pad_len = args.max_seq_length - len(input_ids)
        mask = torch.tensor([1] * args.max_seq_length)
        if pad_len > 0:
            input_ids = input_ids + [tokenizer.vocab['<pad>']] * pad_len
            mask[-pad_len:] = 0

        all_input_ids.append(input_ids)
        all_mask.append(mask.tolist())
        if args.regression:
            all_labels.append([float(examples[i].label)])
        else:
            all_labels.append(int(examples[i].label))

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_mask, dtype=torch.long)

    all_labels = torch.tensor(all_labels, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
    return dataset

# 将一个输入文件转换为tsv格式
# 将输入文件fname中的每一行数据的第一列提取出来，并在每行的末尾添加一个制表符和数字lable
def convert_file(fname):
    synth_seq = []
    fid = open(fname)
    synth = fid.read()
    synth = synth.split('\n')
    fid.close()
    fid = open("tmp/dev.tsv", "w")
    fid.write("\n")
    for seq in synth:
        if not (seq == ""):
            line = seq.split(",")[0] + "\t0"
            fid.write(line + "\n")

# 基于原本的文件转换函数，新增一个参数，可以指定输出tsv文件的文件名
# 将输入文件fname中的每一行数据的第一列提取出来，并在每行的末尾添加一个制表符和数字lable
def convert_file_parameter(fname, outputname, test=False):
    # outputname可以选择为train、dev
    synth_seq = []
    fid = open(fname)
    synth = fid.read()
    synth = synth.split('\n')
    fid.close()
    fid = open("tmp/"+outputname+".tsv","w")
    fid.write("\n")
    for seq in synth:
        if not(seq==""):
            if test:    # 评估模式时，只需要提取序列数据，其标签手动设置为0
                split_seq = seq.split(",")
                seq = split_seq[0].strip()
                seq = seq + ',0'    # 评估的数据为人工序列
            line = seq.replace(",","\t")
            fid.write(line+"\n")

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(  # 指定输入数据文件的路径
        "--synth_file",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(  # 选择预训练模型的类型
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(  # 指定预训练模型的路径或快捷名称
        "--model_name_or_path",
        default='bert_base',
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(  # 指定训练任务的名称，任务的名称应该是一个预定义的任务名称，可以在 processors 字典中找到。
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(  # 指定输出结果的目录路径，训练模型的预测结果和检查点文件将被写入这个目录。
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(  # 指定经过分词后的序列的最大长度
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")  # 表示是否进行训练
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")  # 表示是否在开发集上进行评估
    parser.add_argument("--evaluate_during_training", action="store_true", help="Whether to run eval on the dev set.")  # 表示是否在开发集上进行评估
    parser.add_argument(  # 每个GPU或CPU上的训练批次大小
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(  # 每个GPU或CPU上的评估批次大小
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(  # 总的训练周期数
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    # 每隔多少个训练步骤打印一次训练日志
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    # 每隔多少个训练步骤保存一次模型检查点
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(  # 覆盖输出目录的内容
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(  # 是否进行回归任务
        "--regression", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(  # 如果大于 0，设置总的训练步数来代替训练周期数。如果设置了该参数，将覆盖 num_train_epochs。
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(  # 指定累积梯度的步数，即在执行反向传播/更新之前累积的步数
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # parser.add_argument("--disc_score_output_dir", default=None, type=str, help="the output dir of disc score fie")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower() # 将任务名称转换为小写
    if args.regression:
        print('Task not found, assuming it to be sst-2')
        processor = processors['sst-2']()
        args.output_mode = "Rergression"
        num_labels = 1
    else:
        print('Task not found, assuming it to be sst-2')
        processor = processors['sst-2']()
        args.output_mode = output_modes['sst-2']
        label_list = processor.get_labels()
        num_labels = len(label_list)

    # 加载预训练的模型和分词器
    args.model_type = args.model_type.lower()  # 模型类型，转小写

    # 从预训练模型 bert-base 的配置创建一个 ProteinBertConfig 对象
    config = ProteinBertConfig.from_pretrained("bert-base")

    config.hidden_dropout_prob = args.dropout
    config.attention_probs_dropout_prob = args.dropout
    # model_name_or_path参数：指定预训练模型的路径或快捷名称
    model = ProteinBertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    # 创建一个 TAPETokenizer 分词器对象，使用 'iupac' 词汇表来处理蛋白质序列
    tokenizer = TAPETokenizer(vocab='iupac')

    model.to(args.device)

    # 一个临时的data文件夹，tmp
    args.data_dir = "tmp"

    # convert_file(args.synth_file)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # 训练数据集，test=false表明不使用测试集，evaluate=false表明不打开评价模式
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, test=False, evaluate=False)
        train(args, train_dataset, model, tokenizer)


    # Evaluation
    results = {}
    if args.do_eval:
        # 打分
        scores, _ = evaluate(args, model, tokenizer, test=True, prefix="")
        # 打开数据文件
        fid = open(args.synth_file)
        text = fid.read()  # 读取原始数据文件的内容
        lines = text.split('\n')    # 每行用\n来分隔
        fid.close()

        parts = os.path.splitext(args.synth_file)   # 创建输出文件的名称：xxx_disc.txt
        fid = open(parts[0] + "_disc" + parts[1], "w")
        for i in range(0, len(lines) - 1):
            line = lines[i]
            # 将原始数据行内容与对应的评估分数拼接成新的内容，写入新文件中。使用逗号","将两部分内容分隔。
            fid.write(line + "," + str(scores[i]) + "\n")


if __name__ == "__main__":
    main()
