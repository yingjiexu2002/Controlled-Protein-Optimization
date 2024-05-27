#set GLUE_DIR to /path/to/glue
#export GLUE_DIR=/export/share/akhilesh-gotmare/transformers/glue_data
export GLUE_DIR=/export/home/ProGen/discriminators/fluorescence/protein_tasks
export TASK_NAME=data_adv

## if training on a single GPU, remove
#  -m torch.distributed.launch \
#     --nproc_per_node 8
# from the below command
# results might differ to a slight extent, since the final evaluators will be different due to different effective batchsize used

#final RoBERTa model saved at --disc_bert_output


# CUDA_VISIBLE_DEVICES=2 python get_disc_score.py --model_type bert \
#     --model_name_or_path disc_bert_output1/checkpoint-20000/pytorch_model.bin \  # 在训练时，使用bert-base; 在评估时，使用自己的模型路径
#     --task_name SST-2 \ 
#     --dropout 0.1 \ 
#     --do_eval \     # 执行main函数中的evaluation
#     --synth_file train_test/dev.txt\    # 在训练模式中，需要设置为训练集的路径。在evaluation模式中，需要设置为评估集的路径
#     --max_seq_length 256 \
#     --per_gpu_train_batch_size 16 \
#     --learning_rate 1e-5 \
#     --num_train_epochs 3.0 \
#     --output_dir disc_bert_output1 \    # 输出路径
#     --overwrite_output_dir \
#     --save_steps 5000 \
#     --logging_steps 50 \
#     --do_train


CUDA_VISIBLE_DEVICES=3 python get_disc_score_simplified.py --model_type bert \
    --model_name_or_path disc_bert_output1/checkpoint-20000/pytorch_model.bin \
    --task_name SST-2 \
    --dropout 0.1 \
    --do_eval \
    --synth_file train_test/dev.txt \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size 16 \
    --output_dir disc_bert_output1 \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --save_steps 5000 \
    --logging_steps 50 \
