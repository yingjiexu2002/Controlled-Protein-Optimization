version="5"
for i in {0..4}
do
    # 输出当前正在处理的code
    echo "Generating code ${i}"
    CUDA_VISIBLE_DEVICES=0 python3 batch_lysozyme_gen.py --code ${i} \
        --fname data/${version}_samples_progen \
        --num_sample_batches 320 \
        --batch_size 32 \
        --gen_length 512 \
        --top_p 0.75 \
        --top_k 0 \
        --rep_penalty 1.2 \
        --pt_model ../pro-gen_training/checkpoints/progen/my_progen_finetuned.pt \

done