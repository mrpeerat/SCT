#!/bin/bash  

queue_sizes=(131072)
learning_rate_all=(1e-4) 
teacher_temps=(0.04) 
student_temps=(0.05) 

train_batch_size=128
max_seq_langth=128
seed=1000

student_model_name_or_path='mrp/SCT_BERT-Small' # using the SCT model as the student model
train_dataset_path='data/back_translated_nli.txt' # training data from: https://drive.google.com/file/d/19O2NArJz_RlVNNGRbBnnWxNMW-7HaFZ8/view
dev_dataset_path='data/sts-dev.tsv'
gpu_device=0

for teacher_temp in "${teacher_temps[@]}"
do
    for student_temp in "${student_temps[@]}"
    do
        for queue_size in "${queue_sizes[@]}"
        do
            for learning_rate in "${learning_rate_all[@]}"
            do
                python SCT_main_distillation.py \
                    --model_save_path experiments/SCT_Distillation_BERT_Small_bs${train_batch_size}_qs${queue_size}_t${teacher_temp}_s${student_temp}_lr${learning_rate}_Seed${seed} \
                    --student_model_name_or_path ${student_model_name_or_path} \
                    --train_dataset_path ${train_dataset_path} \
                    --dev_dataset_path ${dev_dataset_path} \
                    --train_batch_size ${train_batch_size} \
                    --inference_batch_size ${train_batch_size} \
                    --eval_batch_size ${train_batch_size} \
                    --max_seq_length ${max_seq_langth} \
                    --num_epochs 20 \
                    --learning_rate ${learning_rate} \
                    --teacher_temp ${teacher_temp} \
                    --student_temp ${student_temp} \
                    --queue_size ${queue_size} \
                    --gpu_device ${gpu_device} \
                    --seed ${seed}
            done
        done
    done
done
