#!/bin/bash

# Set how many GPUs to use

NUM_GPU=1

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)


# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \
torchrun --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path /root/autodl-tmp/roberta-large \
    --sbert_model_path /root/autodl-tmp/all-MiniLM-L12-v1/ \
    --train_file  /root/dataset/train_data_ernie_processed.csv \
    --output_dir  /root/sim-out/roberta-large-finetune \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps=2 \
    --learning_rate 5e-5 \
    --max_seq_length 256 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --seed 32 \
    --mlm_weight 0.05 \
    --mlm_probability 0.1 \
    "$@"
