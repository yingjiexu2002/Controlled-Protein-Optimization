# model_dir: 模型输出路径
# model_path：预训练模型路径
# pklpath: 蛋白质序列pickle格式文件的路径
# 预训练模型名称：pretrain_progen_full.pth
# 微调后的模型名称：progen_finetuned_on_lysozymes.pth
# CUDA_VISIBLE_DEVICES=7 python train_progen.py --model_dir checkpoints/progen/ \
# 改为多卡训练
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_progen.py --model_dir checkpoints/progen/ \
    --model_path checkpoints/progen_finetuned_on_lysozymes.pth \
    --seed 9 \
    --sequence_len 511 \
    --num_epochs 10 \
    --num_layers 36 \
    --batch_size 2 \
    --warmup_iteration 15000 \
    --save_iter 5000 \
    --pklpath dataset/train_data[60,512].p \
    --log_interval 100 \
    --learning_rate 0.0001 \