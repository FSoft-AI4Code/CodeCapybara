CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29450 --nnodes=1 --nproc_per_node=gpu main/train.py --train-batch-size 2 --val-batch-size 4 --num-workers 2 --config-path configs/config.yml --model-type lora --use-wandb 1