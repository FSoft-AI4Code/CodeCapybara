import os
import re
import json
import gzip
import itertools
from typing import Union, List
import numpy as np

def create_filepath(filepath):
    if os.path.exists(filepath):
        filename, ext = os.path.splitext(filepath)
        file_index = re.search(r"\d+$", filename)
        if file_index is None:
            filename = "{}1".format(filename)
            file_index = 1
        else:
            file_index = int(file_index.group(0))
            file_index = int(file_index)
        while os.path.exists("{}{}".format(filename, ext)):
            filename = filename[:-len(str(file_index))] + str(file_index+1)
            file_index += 1
        filepath = "{}{}".format(filename, ext)
    return filepath

def create_mbpp_instruction(example):
    description = example["text"]
    test_cases = example["test_list"]
    prompt = "You are an expert Python programmer, and here is your task: {description} Your code should pass these tests:\n\n{tests}\n"
    instruction = prompt.format(description=description, tests="\n".join(test_cases))
    return instruction

def load_data(dataset_name):
    data_path_mapping = {
            "mbpp": "./data/mbpp.jsonl",
            "humaneval": "./data/HumanEval.jsonl.gz"
            }
    data_path = data_path_mapping[dataset_name]
    data = []
    if data_path.endswith(".jsonl.gz"):
        with gzip.open(data_path, "rt") as f:
            data = [json.loads(line) for line in f]
    elif data_path.endswith(".json"):
        with open(data_path, "r") as f:
            data = json.load(f)
    else:
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]

    instructions = []
    task_ids = []
    if dataset_name == "mbpp":
        data = list(filter(lambda x: x["task_id"] in range(11, 511), data))
        instructions = list(map(create_mbpp_instruction, data))
        task_ids = list(map(lambda x: x["task_id"], data))
    else:
        task_ids = [ex["task_id"] for ex in data]
        instructions = [ex["prompt"] for ex in data]

    return task_ids, instructions


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
