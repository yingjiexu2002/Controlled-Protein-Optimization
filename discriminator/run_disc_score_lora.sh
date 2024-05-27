# synth_file：待打分的序列文件
family_list=('PF00959' 'PF01832' 'PF05838' 'PF06737' 'PF16754')
# family_list=('PF00959')
num=5000
dir="unscored"
modeldir="checkpoint-40000"
seqsize=512
for family in "${family_list[@]}"
do
    CUDA_VISIBLE_DEVICES=3 python get_disc_score_simplified.py --model_type bert \
        --model_name_or_path disc_bert_output1/${modeldir}/pytorch_model.bin \
        --task_name SST-2 \
        --dropout 0.1 \
        --do_eval \
        --synth_file data_score/lora_generation/${dir}/lora_${family}_select${num}.txt \
        --max_seq_length ${seqsize} \
        --per_gpu_eval_batch_size 32 \
        --output_dir disc_bert_output1 \
        --learning_rate 1e-5 \
        --num_train_epochs 3.0 \
        --save_steps 5000 \
        --logging_steps 50 \

    CUDA_VISIBLE_DEVICES=3 python get_disc_score_simplified.py --model_type bert \
        --model_name_or_path disc_bert_output2/${modeldir}/pytorch_model.bin \
        --task_name SST-2 \
        --dropout 0.1 \
        --do_eval \
        --synth_file data_score/lora_generation/${dir}/lora_${family}_select${num}_disc.txt \
        --max_seq_length ${seqsize} \
        --per_gpu_eval_batch_size 32 \
        --output_dir disc_bert_output2 \
        --learning_rate 1e-5 \
        --num_train_epochs 3.0 \
        --save_steps 5000 \
        --logging_steps 50 \

    CUDA_VISIBLE_DEVICES=3 python get_disc_score_simplified.py --model_type bert \
        --model_name_or_path disc_bert_output3/${modeldir}/pytorch_model.bin \
        --task_name SST-2 \
        --dropout 0.1 \
        --do_eval \
        --synth_file data_score/lora_generation/${dir}/lora_${family}_select${num}_disc_disc.txt \
        --max_seq_length ${seqsize} \
        --per_gpu_eval_batch_size 32 \
        --output_dir disc_bert_output3 \
        --learning_rate 1e-5 \
        --num_train_epochs 3.0 \
        --save_steps 5000 \
        --logging_steps 50 \


    if [ -f "data_score/lora_generation/${dir}/lora_${family}_select${num}_disc.txt" ]; then  
        # 删除文件  
        rm "data_score/lora_generation/${dir}/lora_${family}_select${num}_disc.txt"  
        echo "File lora_${family}_select${num}_disc.txt has been deleted."  
    else  
        echo "File lora_${family}_select${num}_disc.txt does not exist."  
    fi

    # 检查文件是否存在  
    if [ -f "data_score/lora_generation/${dir}/lora_${family}_select${num}_disc_disc.txt" ]; then  
        # 删除文件  
        rm "data_score/lora_generation/${dir}/lora_${family}_select${num}_disc_disc.txt"  
        echo "File lora_${family}_select${num}_disc_disc.txt has been deleted."  
    else  
        echo "File lora_${family}_select${num}_disc_disc.txt does not exist."  
    fi
    echo "当前序列，3个鉴别器打分完毕，最终结果在lora_${family}_select${num}_disc_disc_disc.txt"
done

