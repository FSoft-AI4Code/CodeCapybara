import torch
import torch.nn as nn
from typing import List

from models.llama import LLaMA

def sample_top_p(probs, p, num_samples: int = 5):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples = num_samples)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

class CodeCamel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.llama_model = LLaMA(args)
        # for w in self.llama_model.parameters():
        #     w.requires_grad = False
        # self.llama_model.apply(self.llama_model.enable_weights)
        if self.training:
            self.forward = self.generate
    def forward(self, input_ids: torch.Tensor, start_pos: int, mask: torch.Tensor):
        logits = self.llama_model(input_ids, start_pos, mask)
        return logits

    @torch.inference_mode()
    def generate_raw(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        # params = self.model.params
        # assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        # self.tokenizer.pad_id = -1

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(512, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.llama_model.forward(tokens[:, prev_pos:cur_pos], prev_pos, None)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)

            else:
                next_token = torch.argmax(logits, dim=-1)
            
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded
    @torch.inference_mode()
    def generate(
        self,
        prompt_ids,
        max_gen_len: int,
        prompt_lens: torch.Tensor, 
        num_samples: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompt_ids)
        # params = self.model.params
        # assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        self.tokenizer.pad_id = -1

        # prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = torch.min(prompt_lens)
        max_prompt_size = torch.max(prompt_lens)

        total_len = min(1024, max_gen_len + max_prompt_size)

        # tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        # for k, t in enumerate(prompt_ids):
        #     tokens[k, : len(t)] = torch.tensor(t).long()
        if total_len > prompt_ids.shape[1]:
            prompt_ids = nn.functional.pad(prompt_ids, (0, total_len - prompt_ids.shape[1]), value = self.tokenizer.pad_id)
        if num_samples > 1:
            prompt_ids = torch.unsqueeze(prompt_ids, dim = 1).tile(1, num_samples, 1).reshape(prompt_ids.shape[0] * num_samples, -1)

        input_text_mask = prompt_ids != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            if prev_pos == 0:
                logits = self.llama_model.forward(prompt_ids[:1, prev_pos:cur_pos], prev_pos, None)
            else:
                logits = self.llama_model.forward(prompt_ids[:, prev_pos:cur_pos], prev_pos, None)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                if prev_pos == 0:
                    next_token = sample_top_p(probs, top_p, num_samples = num_samples)
                else:
                    next_token = sample_top_p(probs, top_p, num_samples = 1)
            else:
                next_token = torch.argmax(logits, dim=-1)
            
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], prompt_ids[:, cur_pos], next_token
            )
            prompt_ids[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(prompt_ids.tolist()):
            # cut to max gen len
            t = t[: len(prompt_ids[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            # t = t[prompt_lens[i]:]
            decoded.append(self.tokenizer.decode(t))
        return decoded

