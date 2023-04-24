import os
import re
import string
import json
import time
from datetime import datetime
import random
from functools import partial
random.seed(42)
from tqdm import tqdm
from collections import OrderedDict
# import concurrent.futures
import multiprocessing
import logging
from typing import List, Optional, Sequence, Union
from argparse import ArgumentParser
import multiprocessing
import functools
import openai
from openai import openai_object

from data.prompts.prompt import first_PROMPT, second_PROMPT 
from tokenizer import num_tokens_from_messages

openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OPENAI API key.")
openai.api_key = os.getenv("OPENAI_KEY")

MIN_INS_LEN = 3
MAX_INS_LEN = 1e6
NUM_PROMPT_CODE_SNIPPETS = 3
CODE_LENGTH_THRESHOLD = 4000

def encode_prompt(code_snippets, instructions, mode):
    assert mode in ["1st", "2nd"]
    assert len(instructions) <= len(code_snippets)
    if mode == "1st":
        num_code_snippets = len(code_snippets)
        template = first_PROMPT["template"]
        requirement_dict = first_PROMPT["requirements"]
        requirement_str = ""
        for idx, (k, requirement) in enumerate(requirement_dict.items()):
            if k == "input_output":
                if random.uniform(0, 1) < 0.5:
                    continue
            elif k == "length":
                min_length = random.choice(range(0, 60, 10))
                max_length = min_length + random.choice(range(30, 110, 10))
                requirement = requirement.format(min_length=min_length,max_length=max_length)
            requirement_str += f"{idx + 1}. {requirement}\n"
        prompt = template.format(requirements=requirement_str.strip(), num_code_snippets=num_code_snippets)
        for idx, code_snippet in enumerate(code_snippets):
            code_snippet = code_snippet.strip()
            prompt += f"Code snippet {idx + 1}:\n"
            prompt += code_snippet
            prompt += "\n"
        prompt += f"\n\n###\nList of {num_code_snippets} corresponding problem statements:\n"

        idx = 0
        for idx, instruction in enumerate(instructions):
            prompt += f"Problem statement {idx + 1}:\n"
            prompt += instruction
            prompt += "\n"
        else:
            prompt += f"Problem statement {idx + 1}:"
    else:
        assert len(code_snippets) == 1
        assert len(instructions) == 1
        template = second_PROMPT["template"]
        requirement_dict = second_PROMPT["requirements"]
        requirement_str = ""
        for idx, (k, requirement) in enumerate(requirement_dict.items()):
            requirement_str += f"{idx + 1}. {requirement}\n"
        code_snippet = code_snippets[0]
        instruction = instructions[0]
        prompt = template.format(code=code_snippet.strip(), instruction=instruction.strip(), requirements=requirement_str.strip())

    return prompt

def load_data(data_dir, num_examples):
    languages = ["python", "go", "java", "javascript", "ruby", "php", "c", "cpp", "csharp", "rust"]
    data = dict()
    progress_bar = tqdm(total=8e6)
    idx = 0
    for language in languages:
        data_path = os.path.join(data_dir, language, "train.jsonl")
        if not os.path.exists(data_path):
            data_path = os.path.join(data_dir, language, "final", "extract_function_train.jsonl")
        with open(data_path, "r") as f:
            for j, ex in enumerate(f):
                ex = json.loads(ex)
                if j == num_examples:
                    break
                example_id = ex.get("id", idx)
                data.update({example_id: {
                    "code": ex["code"],
                    "instruction": "",
                    "language": language,
                    }})
                idx += 1
                progress_bar.update(1)

    return data

def post_process_1st_response(response, example_ids):
    if response is None:
        return []
    seperator = "Problem statement \d:"
    response_text = response["choices"][0]["message"]["content"]
    raw_instructions = re.split(seperator, response_text)
    raw_instructions.extend([""]*(len(example_ids) - len(raw_instructions)))
    result = dict()
    for idx, (inst, k) in enumerate(zip(raw_instructions, example_ids)):
        if idx == len(raw_instructions) - 1 and response["choices"][0]["finish_reason"] == "length":
            result.update({k: {"pass": False, "instruction": inst, "reason": "cutoff"}})
            continue
        if len(inst.split()) <= MIN_INS_LEN or len(inst.split()) > MAX_INS_LEN:
            result.update({k: {"pass": False, "instruction": inst, "reason": "length"}})
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
        ]
        if any(find_word_in_string(word, inst) for word in blacklist):
            result.update({k: {"pass": False, "instruction": inst, "reason": "blacklist"}})
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            result.update({k: {"pass": False, "instruction": inst, "reason": "punctuation"}})
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            result.update({k: {"pass": False, "instruction": inst, "reason": "ascii"}})
            continue
        result.update({k: {"pass": True, "instruction": inst, "reason": ""}})
    return result


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def one_call(examples: OrderedDict[dict[str, str]], decoding_kwargs):
    sleep_time = 8
    num_requests = 0
    code_examples = [ex["code"] for ex in examples.values()]
    example_ids = [example_id for example_id in examples.keys()]
    example_instructions = []
    prompt = encode_prompt(code_examples, example_instructions, mode="1st")
    messages = [
            {"role": "system", "content": prompt}
            ]
    call_1st_responses, call_2nd_responses = [], []
    decoding_kwargs["max_tokens"] = 4096 - 512 - num_tokens_from_messages(messages)
    if decoding_kwargs["max_tokens"] < 0:
        for ex in examples.values():
            ex["instruction"] = ""
            ex["pass_1st"] = False
            ex["pass_2nd_reason"] = "exceed_max_tokens"

            ex["gen_code"] = ""
            ex["pass_2nd"] = False
            ex["pass_2nd_reason"] = "not_pass_1st"
        return examples, call_1st_responses, call_2nd_responses, num_requests
    # 1st round
    while True:
        try:
            response_1st = openai.ChatCompletion.create(messages=messages,
                                                        **decoding_kwargs,
                                                        # **shared_kwargs
                                                        )
            break
        except openai.error.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
            if "Please reduce your prompt" in str(e):
                max_tokens = decoding_kwargs.get("max_tokens", 2048)
                decoding_kwargs["max_tokens"]= int(max_tokens * 0.8)
                logging.warning(f"Reducing target length to {max_tokens}, Retrying...")
            else:
                logging.warning("Hit request rate limit; retrying...")
                time.sleep(sleep_time)  # Annoying rate limit on requests.

    response_1st.update({"data_ids": example_ids})
    call_1st_responses.append(response_1st)
    num_requests += 1

    result_1st = post_process_1st_response(response_1st, example_ids)
    for ex_id, result in result_1st.items():
        examples[ex_id].update({
            "pass_1st": result["pass"],
            "pass_1st_reason": result["reason"],
            "instruction": result["instruction"],
            })

    # 2nd round
    for ex_id, ex in examples.items():
        if not ex["pass_1st"]:
            ex["pass_2nd"] = False
            ex["pass_2nd_reason"] = "not_pass_1st"
            continue
        code = ex["code"]
        instruction = ex["instruction"]
        prompt = encode_prompt([code], [instruction], mode="2nd")
        messages = [
                {"role": "system", "content": prompt}
                ]
        decoding_kwargs["max_tokens"] = 4096 - 512 - num_tokens_from_messages(messages)
        if decoding_kwargs["max_tokens"] < 0:
            ex["gen_code"] = ""
            ex["pass_2nd"] = False
            ex["pass_2nd"] = "exceed_max_tokens"
            continue
        while True:
            try:
                response_2nd = openai.ChatCompletion.create(messages=messages,
                                                                **decoding_kwargs)
                break
            except openai.error.OpenAIError as e:
                logging.warning(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    max_tokens = decoding_kwargs.get("max_tokens", 2048)
                    decoding_kwargs["max_tokens"]= int(max_tokens * 0.8)
                    logging.warning(f"Reducing target length to {max_tokens}, Retrying...")
                else:
                    logging.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)  # Annoying rate limit on requests.
        num_requests += 1
        gen_code = response_2nd["choices"][0]["message"]["content"]
        response_2nd.update({"data_id": ex_id})
        call_2nd_responses.append(response_2nd)
        ex["gen_code"] = gen_code
        if response_2nd["choices"][0]["finish_reason"] == "length":
            ex["pass_2nd"] = False
            ex["pass_2nd_reason"] = "cutoff"
            continue
        else:
            ex["pass_2nd"] = True
            ex["pass_2nd_reason"] = ""

    return examples, call_1st_responses, call_2nd_responses, num_requests

def generate_instruction_following_data(
    data,
    request_batch_size=500,
    num_instructions_to_generate=100,
    output_dir="./output_dir",
    model_name="gpt-3.5-turbo",
    temperature=1.0,
    top_p=1.0,
):
    now = datetime.now()
    datetime_string = now.strftime("%Y-%m-%d %H:%M:%S")
    output_dir = os.path.join(output_dir, now.strftime("%Y-%m-%d_%H:%M:%S"))
    print("output_dir = {}".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    count_tokens = 0
    keep = 0
    num_generated_data = 0
    num_requests = 0

    passed_ids = []
    if os.path.exists("./output_dir/passed_ids.txt"):
        with open("./output_dir/passed_ids.txt", "r") as f:
            passed_ids = f.read()
            passed_ids = passed_ids.strip().split(",")
            passed_ids = [int(i) for i in passed_ids]
    for example_id in passed_ids:
        if example_id in data:
            data.pop(example_id)

    progress_bar = tqdm(total=num_instructions_to_generate)

    decoding_kwargs = dict(
                    model=model_name,
                    temperature=temperature,
                    n=1,
                    top_p=top_p,
                )

    idx = 0
    while keep < num_instructions_to_generate and len(data) > 0:
        num_requested_examples = min(request_batch_size * NUM_PROMPT_CODE_SNIPPETS, num_instructions_to_generate - keep)
        batch_examples_ids = random.sample(list(data.keys()), k=num_requested_examples)
        grouped_examples_ids = [batch_examples_ids[i:i+NUM_PROMPT_CODE_SNIPPETS] for i in range(0, num_requested_examples, NUM_PROMPT_CODE_SNIPPETS)]
        grouped_examples = [OrderedDict([(example_id, data[example_id]) for example_id in example_ids]) for example_ids in grouped_examples_ids]
        request_start = time.time()
        with multiprocessing.Pool() as p:
            for result in p.imap(partial(one_call, decoding_kwargs=decoding_kwargs), grouped_examples):
                examples, call_1st_responses, call_2nd_responses, _num_requests = result
                num_requests += _num_requests
                for ex_id, ex in examples.items():
                    if ex["pass_1st"] and ex["pass_2nd"]:
                        keep += 1
                        passed_ids.append(ex_id)
                        progress_bar.update(1)
                    num_generated_data += 1
                    data.pop(ex_id)
                mode = "w" if idx == 0 else "a"
                with open(os.path.join(output_dir, "openai_1st_reponses.jsonl"), mode) as f:
                    for response in call_1st_responses:
                        json.dump(response, f)
                        f.write("\n")
                with open(os.path.join(output_dir, "openai_2nd_reponses.jsonl"), mode) as f:
                    for response in call_2nd_responses:
                        json.dump(response, f)
                        f.write("\n")
                with open(os.path.join(output_dir, "generated_data.jsonl"), mode) as f:
                    for ex_id, example in examples.items():
                        example["id"] = ex_id
                        json.dump(example, f)
                        f.write("\n")
                idx += 1
                for response in call_1st_responses + call_2nd_responses:
                    count_tokens += response["usage"]["total_tokens"]
        request_duration = time.time() - request_start
        print(f"Batch request tooks {request_duration:.2f}s")

    print(f"Generated {num_generated_data} instructions, kept {keep} instructions")
    with open(os.path.join(output_dir, "count_tokens.jsonl"), "a") as f:
        json.dump({"date": datetime_string, "count_tokens": count_tokens, "cost": count_tokens*1e-6}, f)
    with open("output_dir/passed_ids.txt", "w") as f:
        f.write(",".join([str(i) for i in passed_ids]))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed_data_dir", type=str, default="./seed_data")
    parser.add_argument("--output_dir", type=str, default="./output_dir")
    parser.add_argument("--request_batch_size", type=int, default=500)
    parser.add_argument("--num_instructions_to_generate", type=int, default=1000)
    parser.add_argument("--num_examples_per_language", type=int, default=-1)

    return parser.parse_args()

def main():
    args = parse_args()
    print("Loading data ...")
    seed_data = load_data(args.seed_data_dir, args.num_examples_per_language)
    print("Finished loading data.")
    print("Starting generating data ...")
    generate_instruction_following_data(seed_data, 
                                        request_batch_size=args.request_batch_size,
                                        num_instructions_to_generate=args.num_instructions_to_generate,
                                        output_dir=args.output_dir,
                                        )
    print("Finished generating data ...")

if __name__ == "__main__":
    main()
