#! /bin/bash

cd main/

# model inference
export CUDA_VISIBLE_DEVICES=0,1
N_PROCS=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
NUM_ITERATIONS=10

for _ in $(seq $NUM_ITERATIONS);
do
    python -m torch.distributed.run --nprocs ${N_PROCS} generate.py \
        --output_dir ./generation_output_dir/mbpp \
        --dataset_name 'mbpp' \
        --base_model 'decapoda-research/llama-7b-hf' \
        --lora_weights '' \
        --batch_size 1 \
        --num_return_sequences 20 \
        --load_8bit True
done

# Calculating pass@k with k=1,10,80,100
python eval_mbpp.py --prediction_dir ./generation_output_dir/mbpp
