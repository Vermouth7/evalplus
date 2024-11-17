# CUDA_VISIBLE_DEVICES=0 python evalplus/evaluate.py \
#     --model "/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct" --dataset humaneval --backend hf --greedy --i_just_wanna_run \
#     --my_mode 2 \
#     --insert_layers '[3,20]' \
#     --nrmlize \
#     --operator 'linear_comb' \
#     --coef 0.5 \
#     --split_file "/home/chh/repos/my_ctg/instructions/humaneval/humaneval_2steps_llama_2.json" \
#     --discriminator "/data1/chh/my_ctg/classifier/humaneval/logistic_regression_model.pkl"

# CUDA_VISIBLE_DEVICES=0 python evalplus/evaluate.py \
#     --model "/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct" --dataset mbpp --backend hf --greedy --i_just_wanna_run \
#     --my_mode 1 \
#     --insert_layers '[3,20]' \
#     --nrmlize \
#     --operator 'linear_comb' \
#     --coef 0.5 \
#     --split_file "/home/chh/repos/my_ctg/instructions/mbpp/mbpp_2steps_llama.json" \
#     --discriminator "/data1/chh/my_ctg/classifier/humaneval/logistic_regression_model.pkl"

# rm -rf ./evalplus_results

# CUDA_VISIBLE_DEVICES=3 python evalplus/evaluate.py \
#     --model "/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct" --dataset humaneval --backend hf --greedy --i_just_wanna_run \
#     --my_mode 1 \
#     --insert_layers '[3]' \
#     --nrmlize \
#     --operator 'replace' \
#     --split_file "/home/chh/repos/my_ctg/instructions/humaneval/humaneval_2steps_llama_2.json" 

# rm -rf ./evalplus_results/humaneval

# CUDA_VISIBLE_DEVICES=3 python evalplus/evaluate.py \
#     --model "/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct" --dataset humaneval --backend hf --greedy --i_just_wanna_run \
#     --my_mode 2 \
#     --insert_layers '[3]' \
#     --nrmlize \
#     --operator 'replace' \
#     --split_file "/home/chh/repos/my_ctg/instructions/humaneval/humaneval_2steps_llama_2.json" \
#     --discriminator "/data1/chh/my_ctg/classifier/humaneval/logistic_regression_model.pkl"


# CUDA_VISIBLE_DEVICES=3 python evalplus/evaluate.py \
#     --model "/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct" --dataset mbpp --backend hf --greedy --i_just_wanna_run \
#     --my_mode 1 \
#     --insert_layers '[3]' \
#     --nrmlize \
#     --operator 'replace' \
#     --split_file "/home/chh/repos/my_ctg/instructions/mbpp/mbpp_2steps_llama.json" 

rm -rf ./evalplus_results/mbpp

CUDA_VISIBLE_DEVICES=3 python evalplus/evaluate.py \
    --model "/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct" --dataset mbpp --backend hf --greedy --i_just_wanna_run \
    --my_mode 1 \
    --insert_layers '[3]' \
    --nrmlize \
    --operator 'replace' \
    --base_only \
    --split_file "/home/chh/repos/my_ctg/instructions/mbpp/mbpp_2steps_llama.json" \
    --discriminator "/data1/chh/my_ctg/classifier/humaneval/logistic_regression_model.pkl"