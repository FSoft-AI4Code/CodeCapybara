import sys
sys.path.append(".")
sys.path.append("..")

import logging
import random
import numpy as np
import argparse
from types import SimpleNamespace
import yaml
import math
import os
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, barrier, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from prettytable import PrettyTable
from peft import PeftModel
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict
)

import functools
from transformers import LlamaForCausalLM, LlamaTokenizer, get_linear_schedule_with_warmup
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from torch.distributed.fsdp import (
   FullyShardedDataParallel,
   CPUOffload,
   FullStateDictConfig,
   StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)


from data.dataset import CapybaraDataset
from data.loaders import get_code_contests, get_code_capybara, get_code_alpaca

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def display_params(model):
    param_table = PrettyTable(['Name', 'Trainable', 'Shape', '#Params'])
    param_table.align["Name"] = "l"
    param_table.align["Shape"] = "l"
    
    for name, w in model.named_parameters():
        size = torch.tensor(w.shape)
        param_table.add_row([name, w.requires_grad, size.tolist(), size.prod().item()])
    print(param_table)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config-path', default = 'configs/config.yml', type = str, help = 'the path of the config file')
    ap.add_argument('--seed', type = int, default = 0, help = 'seed for training process')
    ap.add_argument('--train-batch-size', default = 10, type = int)
    ap.add_argument('--val-batch-size', default = 10, type = int)
    ap.add_argument('--num-workers', default = 0, type = int)
    ap.add_argument('--model-type', type = str, choices = ['fine-tuning', 'lora'])
    ap.add_argument('--use-wandb', type = int, default = 1)
    return ap.parse_args()

def parse_config(data):
    if type(data) is list:
        return list(map(parse_config, data))
    elif type(data) is dict:
        sns = SimpleNamespace()
        for key, value in data.items():
            setattr(sns, key, parse_config(value))
        return sns
    else:
        return data

def ddp_setup():
    init_process_group(backend = 'nccl')
def cleanup():
    destroy_process_group()


def save_fsdp_checkpoint(base_path, stats, model, optimizer, scheduler = None):
    path = os.path.join(base_path, f"cp_{stats['epoch']}.tar")
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    saved_checkpoint_data = {
        'epoch': stats['epoch'],
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None
    }
    with FullyShardedDataParallel.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
        cpu_state = model.state_dict()
    saved_checkpoint_data['model'] = cpu_state
        
    torch.save(saved_checkpoint_data, path)
def save_ddp_checkpoint(base_path, stats, model, optimizer, scheduler = None):
    path = os.path.join(base_path, f"cp_{stats['epoch']}.tar")
    saved_checkpoint_data = {
        'epoch': stats['epoch'],
        'model': get_peft_model_state_dict(model.module),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None
    }
    torch.save(saved_checkpoint_data, path)

def load_checkpoint(path, device):
    saved_data = torch.load(path, map_location = device)
    return saved_data['epoch'], saved_data['model'], saved_data['optimizer'], saved_data['scheduler']

def train_one_epoch(stats, loader, model, optimizer, scheduler = None):
    dataset_size = len(loader)
    local_rank = stats['rank']
    use_wandb = stats['use_wandb']
    model.train()
    acc_loss, acc_perplexity, count = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    if local_rank == 0:
        bar = tqdm(loader, total = dataset_size)
        for input_ids, output_ids, mask in bar:
            input_ids = input_ids.cuda()
            output_ids = output_ids.cuda()
            mask = mask.cuda()
            optimizer.zero_grad()
            outputs = model(input_ids = input_ids, attention_mask = mask)
            logits = outputs.logits.transpose(1, 2)
            loss = criterion(logits, output_ids)
            loss.backward()
            optimizer.step()
            if scheduler is not None: scheduler.step()
            loss_cpu = loss.detach().cpu().item()
            acc_loss += loss_cpu
            perplexity = math.exp(loss_cpu)
            if local_rank == 0 and use_wandb:
                wandb.log({
                    'loss': loss_cpu,
                    'perplexity': perplexity,
                    'lr': optimizer.param_groups[0]['lr']
                })
            acc_perplexity += perplexity
            count += 1
            bar.set_description(f'loss: {loss_cpu: .4f} perplexity: {perplexity: .4f}')
    else:   
        for input_ids, output_ids, mask in loader:
            input_ids = input_ids.cuda()
            output_ids = output_ids.cuda()
            mask = mask.cuda()
            optimizer.zero_grad()
            outputs = model(input_ids = input_ids, attention_mask = mask, labels = output_ids)
            logits = outputs.logits.transpose(1, 2)
            loss = criterion(logits, output_ids)
            loss.backward()
            optimizer.step()
            if scheduler is not None: scheduler.step()
            loss_cpu = loss.detach().cpu().item()
            acc_loss += loss_cpu
            perplexity = math.exp(loss_cpu)
            acc_perplexity += perplexity
            count += 1
    return acc_loss / count, acc_perplexity / count
 

@torch.no_grad()
def validate_one_epoch(stats, loader, model):
    dataset_size = len(loader)
    local_rank = stats['rank']

    acc_loss, acc_perplexity, count = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    if local_rank == 0:
        bar = tqdm(loader, total = dataset_size)
        for input_ids, output_ids, mask in bar:
            input_ids = input_ids.cuda()
            output_ids = output_ids.cuda()
            mask = mask.cuda()
            outputs = model(input_ids = input_ids, attention_mask = mask)
            logits = outputs.logits.transpose(1, 2)
            loss = criterion(logits, output_ids)
            loss_cpu = loss.cpu().item()
            acc_loss += loss_cpu
            perplexity = math.exp(loss_cpu)
            acc_perplexity += perplexity
            count += 1
    else:
        for input_ids, output_ids, mask in loader:
            input_ids = input_ids.cuda()
            output_ids = output_ids.cuda()
            mask = mask.cuda()
            outputs = model(input_ids = input_ids, attention_mask = mask)
            logits = outputs.logits.transpose(1, 2)
            loss = criterion(logits, output_ids)
            loss_cpu = loss.cpu().item()
            acc_loss += loss_cpu
            perplexity = math.exp(loss_cpu)
            acc_perplexity += perplexity
            count += 1
    return acc_loss / count, acc_perplexity / count

def main():
    args = parse_args()

    ddp_setup()
    set_seed(args.seed)
    
    config = yaml.load(open(args.config_path), Loader = yaml.FullLoader)
    config = parse_config(config)


    start_epoch = -1
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    

    if local_rank == 0:
        if not os.path.exists(config.checkpoint.dir):
            os.makedirs(config.checkpoint.dir)
        if not os.path.exists(config.log.dir):
            os.makedirs(config.log.dir)
    barrier()
    logger = logging.getLogger('Training CodeCapyraba ...\n')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    file_handler = logging.FileHandler(f'{config.log.dir}/{config.log.file}')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    tokenizer = LlamaTokenizer.from_pretrained(config.model.hf_model)
    tokenizer.pad_token_id = 0
    if local_rank == 0:
        logging.info('HUGGING FACE MODEL:', config.model.hf_model)
    model = LlamaForCausalLM.from_pretrained(config.model.hf_model).train()

    model_data, optimizer_data, scheduler_data = [None] * 3
    resume_train = os.path.exists(config.checkpoint.old_checkpoint or '')
    if resume_train:
        if local_rank == 0:
            logger.info('Resume training ...')
        start_epoch, model_data, optimizer_data, scheduler_data = load_checkpoint(config.checkpoint.old_checkpoint, 'cpu')
    elif local_rank == 0:
        logger.info('Train from scratch ...')
    start_epoch += 1


    if args.model_type == 'lora':
        
        lora_config = LoraConfig(
            r = config.model.lora.r,
            lora_alpha = config.model.lora.alpha,
            target_modules = config.model.lora.target_modules,
            lora_dropout = config.model.lora.dropout,
            bias = "none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        if model_data is not None:
            model.load_state_dict(model_data, strict = False)

        display_params(model)
        model = model.cuda()
        model = DDP(model)
        save_checkpoint = save_ddp_checkpoint
    elif args.model_type == 'fine-tuning':
        if local_rank == 0:
            logger.info('[MODEL TYPE] fine-tuning')
            display_params(model)
        if model_data is not None:
            model.load_state_dict(model_data)
        model = model.cuda()
        
        llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
            },
        )
       
        model = FullyShardedDataParallel(
            model,
            auto_wrap_policy=llama_auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=True),
            sync_module_states = True
        )

        save_checkpoint = save_fsdp_checkpoint
    else:
        raise RuntimeError(f"The model type {args.model_type} isn't supported")

    dataset_names = config.datasets

    train_data, val_data = [], []
    for dataset in dataset_names:
        dataset = vars(dataset)
        dataset_name, dataset_path = next(iter(dataset.items()))
        
        if dataset_name == 'code-contest':
            data = get_code_contests(tokenizer, config.max_seq_length)
        elif dataset_name == 'codealpaca':
            data = get_code_alpaca(dataset_path, tokenizer, config.max_seq_length)
        elif dataset_name == 'codecapybara':
            data = get_code_capybara(dataset_path, tokenizer, config.max_seq_length)
        else:
            raise RuntimeError(f"This dataset ({dataset_name}) isn't supported")

        train_size = int(len(data) * 0.8)
        train_data.extend(data[:train_size])
        val_data.extend(data[train_size:])
        if local_rank == 0:
            logger.info(f'LOADED the dataset: {dataset_name}')

   
    train_dataset = CapybaraDataset(train_data,tokenizer, max_seq_length = config.max_seq_length)
    train_sampler = DistributedSampler(train_dataset) 
    train_batch_size = args.train_batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = train_batch_size,
        num_workers = args.num_workers,
        collate_fn = train_dataset.collate,
        sampler = train_sampler
    )

    val_dataset = CapybaraDataset(val_data, tokenizer, max_seq_length = config.max_seq_length)
    val_sampler = DistributedSampler(train_dataset) 
    val_batch_size = args.train_batch_size
    val_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = val_batch_size,
        num_workers = args.num_workers,
        collate_fn = val_dataset.collate,
        sampler = val_sampler
    )
    
    learnable_weights = [w for w in model.parameters() if w.requires_grad]
    optimizer = getattr(torch.optim, config.optimizer.name)(learnable_weights, **vars(config.optimizer.params))
    num_warmup_steps = config.scheduler.num_warmup_steps
    num_training_steps = config.scheduler.num_training_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    if optimizer_data is not None:
        optimizer.load_state_dict(optimizer_data)
    if scheduler_data is not None:
        scheduler.load_state_dict(scheduler_data)
    
    use_wandb = args.use_wandb
    stats = {'epoch': start_epoch, 'rank': local_rank, 'use_wandb': use_wandb}

    
    if local_rank == 0 and use_wandb:
        wandb.init(project = config.wandb.project, name = config.wandb.name)
    for epoch in range(start_epoch, config.epochs):
        if local_rank == 0:
            logger.info(f'EPOCH: {epoch}')
        stats['epoch'] = epoch
        train_loader.sampler.set_epoch(epoch)
        train_loss, train_perplexity = train_one_epoch(stats, train_loader, model, optimizer, scheduler = scheduler)
        logger.info(f'[RANK = {local_rank}] train-loss: {train_loss: .4f} train-perplexity: {train_perplexity: .4f}')
        if local_rank == 0:
            if epoch % config.checkpoint.epochs == 0:
                save_checkpoint(config.checkpoint.dir, stats, model, optimizer, scheduler)
                logger.info(f'SAVED CHECKPOINT AT EPOCH: {epoch}')
        val_loss, val_perplexity = validate_one_epoch(stats, val_loader, model)
        if local_rank == 0 and use_wandb:
            wandb.log({
                'train-loss': train_loss,
                'train-perplexity': train_perplexity,
                'val-loss': val_loss,
                'val-perplexity': val_perplexity
            })
        logger.info(f'[RANK = {local_rank}] val-loss: {val_loss: .4f} val-perplexity: {val_perplexity: .4f}')


    barrier()
    cleanup()
if __name__ == "__main__":
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    main()
