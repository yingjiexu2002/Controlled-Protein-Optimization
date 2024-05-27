version="6"
p=0.75

CUDA_VISIBLE_DEVICES=1 python generate_EC_v2.py --label 2.7.7.6 \
    --fname output_samples/${version}_samples_EC \
    --num_sample_batches 160 \
    --batch_size 32 \
    --gen_length 512 \
    --top_p ${p} \
    --top_k 0 \
    --rep_penalty 1.2 \
    --pt_model checkpoints/lora/EC_finetuned_with_lora.pt \
    --lora_dim 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_model checkpoints/lora/lora_model.210000.pt \

CUDA_VISIBLE_DEVICES=1 python generate_EC_v2.py --label 2.7.7 \
    --fname output_samples/${version}_samples_EC \
    --num_sample_batches 160 \
    --batch_size 32 \
    --gen_length 512 \
    --top_p ${p} \
    --top_k 0 \
    --rep_penalty 1.2 \
    --pt_model checkpoints/lora/EC_finetuned_with_lora.pt \
    --lora_dim 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_model checkpoints/lora/lora_model.210000.pt \

CUDA_VISIBLE_DEVICES=1 python generate_EC_v2.py --label 2.7 \
    --fname output_samples/${version}_samples_EC \
    --num_sample_batches 160 \
    --batch_size 32 \
    --gen_length 512 \
    --top_p ${p} \
    --top_k 0 \
    --rep_penalty 1.2 \
    --pt_model checkpoints/lora/EC_finetuned_with_lora.pt \
    --lora_dim 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_model checkpoints/lora/lora_model.210000.pt \

CUDA_VISIBLE_DEVICES=1 python generate_EC_v2.py --label 2 \
    --fname output_samples/${version}_samples_EC \
    --num_sample_batches 160 \
    --batch_size 32 \
    --gen_length 512 \
    --top_p ${p} \
    --top_k 0 \
    --rep_penalty 1.2 \
    --pt_model checkpoints/lora/EC_finetuned_with_lora.pt \
    --lora_dim 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_model checkpoints/lora/lora_model.210000.pt \