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


CUDA_VISIBLE_DEVICES=4 python get_disc_score_simplified.py --model_type bert \
    --model_name_or_path bert-base \
    --task_name SST-2 \
    --dropout 0.1 \
    --do_train \
    --synth_file train_test/train.txt\
    --max_seq_length 512 \
    --per_gpu_train_batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 5.0 \
    --output_dir disc_bert_output1 \
    --overwrite_output_dir \
    --save_steps 5000 \
    --logging_steps 50 \
    --evaluate_during_training \
    

CUDA_VISIBLE_DEVICES=4 python get_disc_score_simplified.py --model_type bert \
    --model_name_or_path bert-base \
    --task_name SST-2 \
    --dropout 0.1 \
    --do_train \
    --synth_file train_test/train.txt\
    --max_seq_length 512 \
    --per_gpu_train_batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 5.0 \
    --output_dir disc_bert_output2 \
    --overwrite_output_dir \
    --save_steps 5000 \
    --logging_steps 50 \
    --seed 53 \
    --evaluate_during_training \

CUDA_VISIBLE_DEVICES=4 python get_disc_score_simplified.py --model_type bert \
    --model_name_or_path bert-base \
    --task_name SST-2 \
    --dropout 0.1 \
    --do_train \
    --synth_file train_test/train.txt\
    --max_seq_length 512 \
    --per_gpu_train_batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 5.0 \
    --output_dir disc_bert_output3 \
    --overwrite_output_dir \
    --save_steps 5000 \
    --logging_steps 50 \
    --seed 64 \
    --evaluate_during_training \