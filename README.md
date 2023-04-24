# CodeCapybara: Code Instruction Tuning

We introduce CodeCapybara - A Code specialized Instruction-following Large Language Model.

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Data Release]()
- [Checkpoint Release]()
- [Installation](#installation)
- [Instruction Tuning](#instruction-tuning)
- [Benchmarking](#benchmarking)
- [Example Outputs](#example-outputs)
- [Future Plans](future-plan)
- [Contributing](#contributing)
- [License](#license)

## Overview
We follow several recent techniques of instruction tuning to collect data and train an instruction-following model with ability to generate executable code from human language description.

We can divide our process for training CodeyCapybara into two stages:
1. **Data Collection**: We collect data generated through OpenAI `gpt-3.5-turbo` as well as code generation supervised dataset.
2. **Instruction Tuning**: We fine-tune our model from MetaAI's LLaMA checkpoint with parameter-efficient fine-tuning methods.

### Data Collection
In this stage, we follow previous works to collect instruction data. To ensure the quality of the code data used in the fine-tuning stage, we make some modifications from data Self-Instruct data generation procedure.
| Data source | No. samples |
|-|-|
|Only Instruction Generation| 20,574|
|CodeAlpaca| 20,022 |
|DeepMind's Code Contests| 13,328 |
| **Total**| **53,924**|

#### Only Instruction Generation
To ensure the code quality, which will be later used as target in the fine-tuning step,  we leverage an unsupervised dataset that only contains code snippets crawled from open-sources. We then design a prompt to ask ChatGPT to generate a corresponding instruction for each code snippet. In other words, to obtain a pair (Instruction-Output), we ask ChatGPT to generate the instruction given the output as human written code snippet.

Our unsupervised dataset contains code functions that covers a wide range of programming problem in 10 programming languages, i.e `Python, Javascript, Java, Golang, Ruby, Rust, PHP, C, C++, C#`

We obtains our dataset through `gpt-3.5-turbo` OpenAI API. Each instruction-output pair is generated through 2 rounds of API calling.
	- In 1st round, we include a code function (i.e output) in the prompt, and ask `gpt-3.5-turbo` to generate a corresponding instruction.
	- In 2nd round, since the code function does not guarantee an executable program, we include both 1st round generated instruction and code function to a new prompt and ask model to generate an executable program with libraries imported and dependencies implementation along with the given code function.
 
- Our prompt template can be found [here](./data/prompts/prompt.py).
- Our script for 2 rounds data generation can be found [here](./data_generation/data_generation.py).

#### [Code Alpaca](https://github.com/sahil280114/codealpaca)
For the second source of data, our intention is to follow [Self-Instruct](https://arxiv.org/abs/2212.10560) paper to completely generate various code problems in the format of (Instruction-Input-Output) data from a seed dataset.

We reuse the generated instruction data from [Code Alpaca](https://github.com/sahil280114/codealpaca/blob/master/data/code_alpaca_20k.json) to reduce API calling cost since what they did is similar to our purpose.

#### [DeepMind's Code Contests](https://github.com/deepmind/code_contests)
We also leverage the supervised code generation dataset. There are various code generation dataset with high quality and quantity, such as APPS (5,000 datapoints in train split), MBPP (500 datapoints in train split).

In this version, we select [DeepMind's Code Contests](https://github.com/deepmind/code_contests) dataset, which contains competitive programming problems with detailed description and test cases. The train split we employ to fine-tune our model contains approximately 13,000 datapoints.

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
| Alpaca-LoRA |  decapoda-research/llama-7b-hf | 8.00 | 10.00 | 10.37|
| CodeCapybara-LoRa |  decapoda-research/llama-7b-hf | 9.61 | 11.62 | 12.02 |
| CodeCapybara |  | **11.10** | **13.33** | **13.41** |

### MBPP
| Model |Base checkpoint | pass@1 | pass@10 | pass@100 |
| ------- | ----| ------- | ------- | -------|
| LLaMA |  decapoda-research/llama-7b-hf | **17.97** | **21.96**| **23.13**|
| Alpaca-LoRA |  decapoda-research/llama-7b-hf | 12.73 | 15.92 | 16.87 |
| CodeCapybara-LoRa |  decapoda-research/llama-7b-hf | 13.11| 17.85| 19.22 |
| CodeCapybara| | | | |

## Data Release

## Checkpoint Release

## Installation

```bash
conda create -n codecapybara -y
conda activate codecapybara
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

$${pass@k} = \underset{\text { Problems }}{\mathbb{E}}\left[1-\frac{C^{k}_{n-c}}{C^{k}_{n}}\right]$$

Here we choose `n = 200` as employed in the paper.

### MBPP
Replacing the `humaneval` by `mbpp`
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

## Example Outputs

## Future Plans

## Contributing

## License

Feel free to cite us
```bibtex
@misc{codecapybara,
	title = {CodeCapybara: Code Instruction Tuning},
	author = {},
	year = {2023},
}
```
