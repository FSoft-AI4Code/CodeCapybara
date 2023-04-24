import os
import sys
import re
from tqdm import tqdm
import json
import math

import fire
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import PeftModel, get_peft_model, LoraConfig
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter
from utils.evaluation import create_filepath, load_data

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if  torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def main(
        output_dir: str = "./output_dir",
        dataset_name: str = "mbpp",
        load_8bit: bool = False,
        base_model: str = "",
        tokenizer: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        prompt_template: str = "",
		batch_size: int = 64,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_return_sequences: int = 1,

        ):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    os.makedirs(output_dir, exist_ok=True)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp = world_size != 1

    if ddp:
        # init distributed process group
        dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=local_rank,
                )
        torch.cuda.set_device(local_rank)
    device_map = {"": local_rank}

    prompter = Prompter(prompt_template)
    if tokenizer is None or tokenizer == "":
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer)
    if lora_weights is None or lora_weights == "":
        model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map=device_map,
                )
    elif lora_weights.endswith(".bin"):
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",

        )
        model = get_peft_model(model, config)

        state_dict = torch.load(lora_weights, map_location="cuda:0")
        pretrained_dict = {k: v for k, v in model.state_dict().items() if k not in state_dict.keys()}
        state_dict.update(pretrained_dict)
        model.load_state_dict(state_dict)
    else:
        model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map=device_map,
                )
        model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map=device_map,
                )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    tokenizer.padding_size = "left"

    if not load_8bit:
        model.haft()

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # loading data
    task_ids, instructions = load_data(dataset_name)
    if dataset_name == "humaneval":
        prompts = instructions
    else:
        prompts = [prompter.generate_prompt(instruction) for instruction in instructions]

    num_samples_this_rank = math.ceil(len(prompts)/world_size)
    _start_idx = num_samples_this_rank*local_rank
    _end_idx = num_samples_this_rank*(local_rank + 1)
    prompts = prompts[_start_idx:_end_idx]
    task_ids = task_ids[_start_idx:_end_idx]
    print("Rank: {}, task_ids: {}".format(local_rank, task_ids))

    output_strings = []
    for idx in tqdm(range(0, len(prompts), batch_size), desc="Rank {}".format(local_rank)):
        batch_prompts = prompts[batch_size*idx:batch_size*(idx+1)]
        # tokenization
        inputs = tokenizer(batch_prompts, 
                           truncation=False,
                           padding=False,
                           )
        input_ids = inputs["input_ids"]
        batch_max_length = max(len(_input_ids) for _input_ids in input_ids)
        new_input_ids, attention_mask = [], []
        for _input_ids in input_ids:
            padding_size = batch_max_length - len(_input_ids)
            new_input_ids.append([tokenizer.pad_token_id]*padding_size + _input_ids)
            attention_mask.append([False]*padding_size + [True]*len(_input_ids))
        input_ids = torch.LongTensor(new_input_ids)
        input_ids = input_ids.to(device)
        attention_mask = torch.BoolTensor(attention_mask)
        attention_mask = attention_mask.to(device)

        this_batch_size = input_ids.shape[0]

        try:
            generation_config = GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    num_return_sequences=num_return_sequences,
                    )

            with torch.no_grad():
                output_ids = model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_config=generation_config,
                        return_dict_in_generate=False,
                        max_new_tokens=128,
                        )
        except torch.cuda.OutOfMemoryError:
            print("Rank {} out of memory ... continue ...".format(local_rank))
            torch.cuda.empty_cache()
            output_strings.extend([""]*(this_batch_size*num_return_sequences))
            continue
        batch_output_strings = [tokenizer.decode(s,
                                           skip_special_tokens=True,
                                           ignore_tokenization_space=True)
                          for s in output_ids]
        output_strings.extend(batch_output_strings)

    filepath = create_filepath(os.path.join(output_dir, "generation.jsonl"))
    with open(filepath, "w") as f:
        for j, output_str in enumerate(output_strings):
            if output_str == "":
                continue
            task_id = task_ids[j//num_return_sequences]
            if dataset_name == "mbpp":
                output_str = prompter.get_response(output_str)
                json.dump({"task_id": task_id, "trg_prediction": output_str, "rank": local_rank}, f)
            elif dataset_name == "humaneval":
                json.dump({"task_id": task_id, "completion": output_str, "rank": local_rank}, f)
            f.write("\n")

    print("Rank {} finished. Predictions saved to {}".format(local_rank, filepath))
 
if __name__ == "__main__":
    fire.Fire(main)

