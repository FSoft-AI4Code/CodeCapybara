import torch

def get_subsequent_mask(max_seq_len):
    mask = torch.full((max_seq_len, max_seq_len), float('-inf'))
    mask = torch.triu(mask, diagonal = 1)
    return mask