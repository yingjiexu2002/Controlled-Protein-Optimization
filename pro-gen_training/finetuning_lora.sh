# model_dir: 模型输出路径
# model_path：预训练模型路径
# pklpath: 蛋白质序列pickle格式文件的路径
# CUDA_VISIBLE_DEVICES=2 python train_lora.py --model_dir checkpoints/lora/ \
CUDA_VISIBLE_DEVICES=1,2,4,5 python -m torch.distributed.launch --nproc_per_node=4 train_lora.py --model_dir checkpoints/lora/ \
    --model_path checkpoints/progen/my_progen_finetuned.pt \
    --seed 43 \
    --sequence_len 511 \
    --num_epochs 10 \
    --num_layers 36 \
    --batch_size 4 \
    --warmup_iteration 15000 \
    --save_iter 5000 \
    --pklpath dataset/train_data[60,512].p \
    --lora_dim 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --log_interval 100 \
    --learning_rate 0.0001 \