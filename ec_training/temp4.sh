version="4"

CUDA_VISIBLE_DEVICES=3 python generate_EC.py --label 2.7.7.6 \
    --fname output_samples/${version}_samples_EC \
    --num_sample_batches 320 \
    --batch_size 32 \
    --gen_length 512 \
    --top_p 0.5 \
    --top_k 0 \
    --rep_penalty 1.2 \
    --pt_model checkpoints/lora/EC_finetuned_with_lora.pt \
    --lora_dim 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_model checkpoints/lora/lora_model.235000.pt \