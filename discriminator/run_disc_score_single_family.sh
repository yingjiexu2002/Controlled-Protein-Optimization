# synth_file：待打分的序列文件
num=2000
dir="single_family"
type="lora"
family="PF00959"
step_list=(0 2000 4000 6000 8000)
for step in "${step_list[@]}"
do
    CUDA_VISIBLE_DEVICES=3 python get_disc_score_simplified.py --model_type bert \
        --model_name_or_path disc_bert_output1/checkpoint-20000-v1/pytorch_model.bin \
        --task_name SST-2 \
        --dropout 0.1 \
        --do_eval \
        --synth_file data_score/${type}_generation/${dir}/${type}_${family}_select${num}_${step}.txt \
        --max_seq_length 256 \
        --per_gpu_eval_batch_size 16 \
        --output_dir disc_bert_output1 \
        --learning_rate 1e-5 \
        --num_train_epochs 3.0 \
        --save_steps 5000 \
        --logging_steps 50 \

    CUDA_VISIBLE_DEVICES=3 python get_disc_score_simplified.py --model_type bert \
        --model_name_or_path disc_bert_output2/checkpoint-20000-v1/pytorch_model.bin \
        --task_name SST-2 \
        --dropout 0.1 \
        --do_eval \
        --synth_file data_score/${type}_generation/${dir}/${type}_${family}_select${num}_${step}_disc.txt \
        --max_seq_length 256 \
        --per_gpu_eval_batch_size 16 \
        --output_dir disc_bert_output2 \
        --learning_rate 1e-5 \
        --num_train_epochs 3.0 \
        --save_steps 5000 \
        --logging_steps 50 \

    CUDA_VISIBLE_DEVICES=3 python get_disc_score_simplified.py --model_type bert \
        --model_name_or_path disc_bert_output3/checkpoint-20000-v1/pytorch_model.bin \
        --task_name SST-2 \
        --dropout 0.1 \
        --do_eval \
        --synth_file data_score/${type}_generation/${dir}/${type}_${family}_select${num}_${step}_disc_disc.txt \
        --max_seq_length 256 \
        --per_gpu_eval_batch_size 16 \
        --output_dir disc_bert_output3 \
        --learning_rate 1e-5 \
        --num_train_epochs 3.0 \
        --save_steps 5000 \
        --logging_steps 50 \


    if [ -f "data_score/${type}_generation/${dir}/${type}_${family}_select${num}_${step}_disc.txt" ]; then  
        # 删除文件  
        rm "data_score/${type}_generation/${dir}/${type}_${family}_select${num}_${step}_disc.txt"  
        echo "File ${type}_${family}_select${num}_${step}_disc.txt has been deleted."  
    else  
        echo "File ${type}_${family}_select${num}_${step}_disc.txt does not exist."  
    fi

    # 检查文件是否存在  
    if [ -f "data_score/${type}_generation/${dir}/${type}_${family}_select${num}_${step}_disc_disc.txt" ]; then  
        # 删除文件  
        rm "data_score/${type}_generation/${dir}/${type}_${family}_select${num}_${step}_disc_disc.txt"  
        echo "File ${type}_${family}_select${num}_${step}_disc_disc.txt has been deleted."  
    else  
        echo "File ${type}_${family}_select${num}_${step}_disc_disc.txt does not exist."  
    fi
    echo "当前序列，3个鉴别器打分完毕，最终结果在${type}_${family}_select${num}_${step}_disc_disc_disc.txt"
done

