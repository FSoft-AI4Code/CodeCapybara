<p align="center" width="100%">
<a  target="_blank"><img src="assets/logo.png" alt="Code-Capybara" style="width: 50%; min-width: 300px; display: block; margin: auto;"></a>
</p>


# CodeCapybara: Open Source LLaMA Model that Follow Instruction-Tuning for Code Generation.

We introduce CodeCapybara - A Code specialized Instruction-following Large Language Model. This repo also attempts to evaluate and reproduce performance results of existing LLMs for code, such as Llama, Alpaca and CodeAlpaca for code generation benchmarks (HumanEval and MBPP).

- ***First attempt to reproduce of LLaMA results*** on widely recognized Code Generation benchmarks
- CodeCapybara is fine-tuned from Llama 7B. Larger models will be available soon.
- We use ***our own dataset in larger scale and more diverse*** to fine-tune Llama under an instruction-tuning style.
- ***Improved evaluation results on HumanEval*** in comparison to LLaMA, Alpaca and CodeAlpaca.
- Full transparency with open source availability: ***all scripts and models are accessible to the community***.
We encourage you to contribute to CodeCapybara and help advance the field of code generation. 

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
To ensure the code quality for later use as targets in the fine-tuning step,  we leverage an unsupervised dataset that only contains code snippets crawled from open-sources. We then design a prompt to ask `gpt-3.5-turbo` to generate a corresponding instruction for each code snippet. In other words, to obtain a pair (instruction-output), we ask `gpt-3.5-turbo` to generate the instruction given the output as human written code snippet.

Our unsupervised dataset contains code functions that covers a wide range of programming problem in 10 programming languages, i.e `Python, Javascript, Java, Golang, Ruby, Rust, PHP, C, C++, C#`

We obtain our dataset through `gpt-3.5-turbo` OpenAI API. Each instruction-output pair is generated through 2 rounds of API calling.
- In 1st round, we include a code function (i.e output) in the prompt, and ask `gpt-3.5-turbo` to generate a corresponding instruction.
- In 2nd round, since the code function does not guarantee an executable program, we include both 1st round generated instruction and code function to a new prompt and ask the model to generate an executable program with libraries imported and dependencies implementation along with the given code function.
 
- Our prompt template can be found [here](./data/prompts/prompt.py).
- Our script for 2 rounds of data generation can be found [here](./data_generation/data_generation.py).

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

We evaluate our models as well as reproduce other models' results on 2 benchmarks, HumanEval and MBPP. All numbers are reported in zero-shot settings.

### HumanEval
| Model |Base checkpoint | pass@1 | pass@10 | pass@100 |
| - | - | - | -  | - |
| LLaMA |  decapoda-research/llama-7b-hf | 10.70| 13.29 | **13.41** |
| LLaMA |  |9.7  | 12.66| 12.80 |
| Alpaca-LoRA |  decapoda-research/llama-7b-hf | 8.00 | 10.00 | 10.37|
| CodeCapybara-LoRa |  decapoda-research/llama-7b-hf | 9.61 | 11.62 | 12.02 |
| CodeCapybara |  | **11.10** | **13.33** | **13.41** |

### MBPP


## Data Release
You can find our used datasets in the folder `data/raw-data`, namely `code_alpaca_20k.json` (from CodeAlpaca) and `generated_data.jsonl` (our own dataset).
## Checkpoint Release

## Installation

```bash
conda create -n codecapybara -y
conda activate codecapybara
```

## Instruction Tuning
We support 2 settings to fine-tune LLaMA models. In the first setting, we refine all the parameters using Fully Sharded Data Parallel, and for the rest, we currently utilize LoRA to adapt the models to the instruction tuning task. You can easily run such settings by the command
```bash
    bash scripts/train.sh
```

which calls `main/train.py`. We also provide some arguments to customize the training process
- --train-batch-size: batch-size of each gpu for training
- --val-batch-size: batch-size of each gpu for validating
- --num-workers: number of workers in the DataLoader
- --config-path: the path of the configuration file. We provide a template in the folder `configs`
- --model-type: setting's used to fine-tune. There are 2 valid values: `fine-tunning` and `lora`.
- --use-wandb: 0 if you don't use *wandb* for logging; otherwise, wandb is used.
Moreover, you can edit the configuration file `configs/config.yml` which contains some notable fields:
- checkpoint
  - dir: the folder contains all the checkpoints
  - old_checkpoint: the path of the old checkpoint. If it is null, the model'll train from scratch; otherwise, it continues training from this checkpoint.
  - epochs: the number of epochs between 2 consecutive model saves.
- epochs: number of epochs for training
- model:
  - hf_model: LLaMA model in HuggingFace format
  - lora: settings for LoRA method
- optimizer: specify optimizer
- scheduler: configurate the hypermeters for a warm-up learning-rate schedule
- max-seq-length: maximum length of the instruction and the response.

## Benchmarking
To evaluate checkpoints on HumanEval or MBPP benchmark, navigate to `main/`
```bash
cd main/
```

We use nucleus sampling for sampling next-token in each prediction step to generate multiple difference code outputs for each problem. Hyperparameter configuration used for our evaluation is specified in the command below.

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
        --load_8bit True \
        --temperature 0.1 \
        --top_p 0.75 \
        --top_k 40
done

# Calculating pass@k with k=1,10,100
python eval_humaneval.py --prediction_dir path/to/prediction/directory
```

`n = NUM_ITERATIONS * batch_size * num_return_sequences`, where `n` is used to estimate `pass@k` as in the [Codex](https://arxiv.org/pdf/2107.03374.pdf) paper.

$${pass@k} = \underset{\text { Problems }}{\mathbb{E}}\left[1-\frac{C^{k}_{n-c}}{C^{k}_{n}}\right]$$

Here we choose `n = 200` as employed in the paper, which results in
- `NUM_ITERATIONS=10`
- `batch_size=1`
- `num_return_sequences=20`

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
        --load_8bit True \
        --temperature 0.1 \
        --top_p 0.75 \
        --top_k 40
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
