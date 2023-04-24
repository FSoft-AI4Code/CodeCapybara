# CodeCapypara: Code Instruction Tuning

We introduce CodeCapypara - A Code specialized Instruction-following Large Language Model.

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Data Release]()
- [Checkpoint Release]()
- [Installation](#installation)
- [Instruction Tuning](#instruction-tuning)
- [Benchmarking](#benchmarking)
- [Future Plans](future-plan)
- [Contributing](#contributing)
- [Example Outputs](#example-outputs)
- [License](#license)

## Overview
We follow several recent techniques of instruction tuning to collect data and train an instruction-following model with ability to generate executable code from human language description.
We can divide our process for training CodeCapypara into two steps:
1. **Data Collection**: We mainly follow Self-Instruct to collect data generated through OpenAI ChatGPT as well as code generation supervised dataset.
2. **Instruction Tuning**: We fine-tune our model from MetaAI's LLaMA checkpoint with parameter-efficient fine-tuning methods.

### Data Collection
In this stage, we follow previous works to collect instruction data. To ensure the quality of the code data used in the fine-tuning stage, we make some modifications from data Alpaca's data generation procedure.
| Data source | No. samples |
|-|-|
|CodeAlpaca| 20,022 |
|Instruction-generation| 20,574|
|DeepMind's Code Contests| 13,328 |
| **Total**| **53,924**|

#### Only Instruction Generation
To ensure the code quality, which will be later used as target in the fine-tuning step,  we leverage an unsupervised dataset that only contains code snippets crawled from open-sources. We then design a prompt to ask ChatGPT to generate a corresponding instruction with each code snippet. In other words, to obtain a pair (Instruction-Output), we ask ChatGPT to generate the instruction given the output as human written code snippet. Our template can be found [here](./data/prompts/prompt.py),
#### [Code Alpaca]()
For the second source of data, we follow [Self-Instruct](https://arxiv.org/abs/2212.10560) paper to generate various code problems in the format of (Instruction-Input-Output) data from a seed dataset.
We reuse the generated instruction data from [CodeAlpaca](https://github.com/sahil280114/codealpaca/blob/master/data/code_alpaca_20k.json) to reduce API calling cost.
#### [DeepMind's Code Contests]()
We also leverage the supervised code generation dataset. There are various code generation dataset with high quality and quantity, such as APPS train split (), MBPP train split (500 datapoints). In this 1st version, we select [DeepMind's Code Contests] dataset, which contains competitive programming problems with detailed description and test cases.
### Instruction Tuning
We tried 2 approaches to fine-tune LLaMA-7B checkpoint on the collected data, including:
- Full-parameter fine-tuning
- HuggingFace's PEFT, same as [AlpacaLoRA](https://github.com/tloen/alpaca-lora#readme)
## Results
We evaluate our models as well as reproducing other models' results on 2 benchmarks, HumanEval and MBPP. All numbers are reported with zero-shot inference.
### HumanEval
| Model |Base checkpoint | pass@1 | pass@10 | pass@100 |
| - | - | - | -  | - |
| LLaMA |  decapoda-research/llama-7b-hf | 10.70| 13.29 | **13.41** |
| LLaMA |  |9.7  | 12.66| 12.80 |
| Alpaca-LoRA |  decapoda-research/llama-7b-hf | 7.56 | 9.14|9.15 |
| CodeCapypara-LoRa |  decapoda-research/llama-7b-hf | 9.61 | 11.62 | 12.02 |
| CodeCapypara |  | **11.10** | **13.33** | **13.41** |
### MBPP
| Model |Base checkpoint | pass@1 | pass@10 | pass@100 |
| ------- | ----| ------- | ------- | -------|
| LLaMA |  decapoda-research/llama-7b-hf | **17.97** | **21.96**| **23.13**|
| Alpaca-LoRA |  decapoda-research/llama-7b-hf | 12.73 | 15.92 | 16.87 |
| CodeCapypara-LoRa |  decapoda-research/llama-7b-hf | 13.11| 17.85| 19.22 |
| CodeCapypara| | | | |

## Data Release

## Checkpoint Release

## Installation

```bash
conda create -n codecapypara -y
conda activate codecapypara
```

## Instruction Tuning
```bash
```
## Benchmarking
To evaluate checkpoints on HumanEval or MBPP benchmark, navigate to `main/`
```bash
cd main/
```
### HumanEval
The first part of the below command generates multiple `.jsonl` files, which will be saved into `path/to/prediction/directory` by inference the model. The command follows after taking predictions as input to calculate pass@k.
```bash
# model inference
export CUDA_VISIBLE_DEVICES=0,1
N_PROCS=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
NUM_ITERATIONS=10

for _ in $(seq $NUM_ITERATIONS);
do
    python -m torch.distributed.run --nprocs ${N_PROCS} generate.py \
        --output_dir path/to/prediction/directory \
        --dataset_name 'humaneval' \
        --base_model 'decapoda-research/llama-7b-hf' \
        --lora_weights '' \
        --batch_size 1 \
        --num_return_sequences 20 \
        --load_8bit True
done

# Calculating pass@k with k=1,10,100
python eval_humaneval.py --prediction_dir path/to/prediction/directory
```

`n = NUM_ITERATIONS * batch_size * num_return_sequences`, where `n` is used to estimate `pass@k` as in the [Codex](https://arxiv.org/pdf/2107.03374.pdf) paper.
$${pass@k} = \underset{\text { Problems }}{\mathbb{E}}\left[1-\frac{\left(\begin{array}{c} n-c \\ k \end{array}\right)}{\left(\begin{array}{l} n \\ k \end{array}\right)}\right]$$

Here we choose `n = 200`.

### MBPP
Replacing the `--dataset_name` with `mbpp`
```bash
# model inference
export CUDA_VISIBLE_DEVICES=0,1
N_PROCS=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
NUM_ITERATIONS=10

for _ in $(seq $NUM_ITERATIONS);
do
    python -m torch.distributed.run --nprocs ${N_PROCS} generate.py \
        --output_dir path/to/prediction/directory \
        --dataset_name 'mbpp' \
        --base_model 'decapoda-research/llama-7b-hf' \
        --lora_weights '' \
        --batch_size 1 \
        --num_return_sequences 20 \
        --load_8bit True
done

# Calculating pass@k with k=1,10,80,100
python eval_mbpp.py --prediction_dir path/to/prediction/directory
```

## Future Plans

## Contributing

## License

Feel free to cite us
```bibtex
@misc{codecapypara,
	title = {CodeCapypara: Code Instruction Tuning},
	author = {},
	year = {2023},
}
```
