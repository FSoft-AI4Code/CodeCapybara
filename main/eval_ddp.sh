#! /bin/bash
PORT=${1:-29500}
# cd ..
export CUDA_VISIBLE_DEVICES=0,1
NUM_PROCS=$(echo $CUDA_VISIBLE_DEVICES | tr ",", "\n" | wc -l)
python -m torch.distributed.run --nproc_per_node=${NUM_PROCS} --master_port ${PORT} generate.py \
    --output_dir ./generation_output_dir \
    --data_path 'data/mbpp.jsonl' \
    --load_8bit True \
    --base_model '/cm/shared/minhnh46/CodeCamel/out-hf/cp1/llama-7b' \
    --tokenizer '/cm/shared/minhnh46/CodeCamel/out-hf/cp1/tokenizer' \
    --lora_weights '' \
    --batch_size 1 \
    --num_return_sequences 12 \
    --prompt_template alpaca
