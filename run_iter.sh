#!/bin/bash

LOG_FILE="output_iter_log.txt"

> "$LOG_FILE"


for layer in {1..32}
do
    #
    echo "Running with insert_layers: [$layer]" | tee -a "$LOG_FILE"

   
    rm -rf ./evalplus_results/mbpp

    CUDA_VISIBLE_DEVICES=2 python evalplus/evaluate.py \
        --model "/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct" \
        --dataset mbpp \
        --backend hf \
        --greedy \
        --i_just_wanna_run \
        --my_mode 1 \
        --insert_layers "[$layer]" \
        --nrmlize \
        --operator 'replace' \
        --base_only \
        --split_file "/home/chh/repos/my_ctg/instructions/mbpp/mbpp_2steps_llama_3.json" >> "$LOG_FILE" 2>&1

done

echo "All tasks completed!" | tee -a "$LOG_FILE"
