# model_dir: 模型输出路径
# model_path：预训练模型路径
# pklpath: 蛋白质序列pickle格式文件的路径
# CUDA_VISIBLE_DEVICES=2 python train_EC.py --model_dir checkpoints/lora/ \
# 改为多卡训练
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 train_EC.py --model_dir checkpoints/lora/ \
    --model_path checkpoints/pretrain_progen_full.pth \
    --seed 43 \
    --sequence_len 511 \
    --num_epochs 40 \
    --num_layers 36 \
    --batch_size 4 \
    --warmup_iteration 15000 \
    --save_iter 5000 \
    --pklpath train_data/train_data_EC1=2_v2.p \
    --lora_dim 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --log_interval 100 \
    --learning_rate 0.0001 \