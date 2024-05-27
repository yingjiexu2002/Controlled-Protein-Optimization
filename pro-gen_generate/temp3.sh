#!/bin/bash
# 循环执行以下代码5次
version="16"
for i in {0..4}
do
    # 输出当前正在处理的code
    echo "Generating code ${i}"
    CUDA_VISIBLE_DEVICES=3 python lora_generation.py --code ${i} \
        --fname data/${version}_samples_lora \
        --num_sample_batches 320 \
        --batch_size 32 \
        --gen_length 512 \
        --top_p 0.75 \
        --top_k 0 \
        --rep_penalty 1.2 \
        --pt_model ../pro-gen_training/checkpoints/lora/progen_finetuned_with_lora.pt \
        --lora_dim 32 \
        --lora_alpha 64 \
        --lora_dropout 0.1 \
        --lora_model ../pro-gen_training/checkpoints/lora/lora_model.41920.pt \

done