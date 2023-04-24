import numpy as np
import json
import jsonlines
import torch
from torch.utils.data import Dataset

from data.utils.pad import pad_batch_2D
from data.utils.mask import get_subsequent_mask
from data.prompts import code_contests_prompts, codealpaca_prompts

import random


class CapybaraDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length = 600):
        self.data = data            
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    def __len__(self):
        return 16#len(self.data)
    def __getitem__(self, index):
        item = self.data[index]
        dataset_name = item['dataset']

        if dataset_name == 'code-contest':
            lang = item['language']
            instruction = item['description']
            output = item['solution']
            prompt = random.choice(code_contests_prompts)
            if prompt.startswith('{'):
                full_input = instruction
            else:
                full_input = prompt.format(language = lang, problem_description = instruction)
            full_input = full_input + ' Output: '
        elif dataset_name == 'codealpaca':
            instruction = item['instruction']
            input = item['input']
            output = item['output']
            if len(input) > 0:
                full_input = codealpaca_prompts['prompt_input'].format(instruction = instruction, input = input)
            else:
                full_input = codealpaca_prompts['prompt_no_input'].format(instruction = instruction)
        elif dataset_name == 'codecapybara':
            full_input = item['instruction'] + ' Output: '
            output = item['gen_code']
        elif dataset_name == 'mbpp':
            full_input = item['text']
            output = item['code']


        prompts = self.tokenizer.encode(full_input)
        targets = self.tokenizer.encode(output)[1:self.max_seq_length - len(prompts)]
        input_ids = prompts + targets
        output_ids = [-100] * (len(prompts) - 1) + targets + [self.tokenizer.eos_token_id]
        return input_ids, output_ids
    def collate(self, batch):
        input_ids, output_ids = list(zip(*batch))
        pad_input_ids = pad_batch_2D(input_ids, value = self.tokenizer.pad_token_id)
        pad_output_ids = pad_batch_2D(output_ids, value = -100)
        pad_input_ids = torch.tensor(pad_input_ids)
        pad_output_ids = torch.tensor(pad_output_ids)
        mask = pad_input_ids.ne(self.tokenizer.pad_token_id)
        return pad_input_ids, pad_output_ids, mask
