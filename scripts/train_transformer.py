import torch
import time
import wandb
import numpy as np
import sys
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import TransformerLM, AdamW
from cs336_basics.utils import get_batch, cross_entropy, save_checkpoint, load_checkpoint, lr_cosine_schedule, gradient_clipping

@dataclass
class ModelConfig:
    vocab_size: int = 10000
    d_model: int = 512
    num_heads: int = 16
    num_layers: int = 4
    d_ff: int = 1344
    rope_theta: int = 10000
    context_length: int = 256

@dataclass
class OptimConfig:
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

@dataclass
class TrainingConfig:
    seed: int = 42
    min_lr: float = 1e-4
    max_lr: float = 1e-3
    max_iters: int = 40000
    warmup_iters: int = 4000
    eval_iters: int = 100
    max_l2_norm: float = 1.0
    log_interval: int = 100
    eval_interval: int = 1000
    checkpoint_interval: int = 10000
    resume_from: str|None = None


@dataclass
class DataConfig:
    batch_size: int = 32
    train_path: str = "../data/tinystories_train.npy"
    eval_path: str = "../data/tinystories_val.npy"
    ckpt_path: str = "../output/ckpt_30000.pt"
    vocab_path: str = "../output/tiny_stores_vocab_train.pkl"
    merges_path: str = "../output/tiny_stores_merges_train.pkl"
    output_dir: str = "../output"

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)

def load_cfg(config_file: str = None):
    cfg = OmegaConf.structured(Config)
    return cfg
    
def load_dataset(input_path:str):
    return np.load(input_path, mmap_mode="r")

def train(cfg):
    wandb.init(
        project="cs336 assignment1",
        name="Train a demo TransformerLM"
    )
    torch.manual_seed(cfg.train.seed)

    device = torch.device("cuda:1")
    output_dir = Path(cfg.data.output_dir)

    dataset_train = load_dataset(cfg.data.train_path)
    dataset_test = load_dataset(cfg.data.eval_path)

    model = TransformerLM(**cfg.model).to(device)
    optimizer = AdamW(params=model.parameters(), **cfg.optim)

    start_iter = 0
    if cfg.train.resume_from:
        print(f"Resuming from checkpoint: {cfg.resume_from}")
        start_iter = load_checkpoint(cfg.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    print("Starting training...")
    start_time = time.time()
    for iteration in tqdm(range(start_iter, cfg.train.max_iters), desc="Training"):
        inputs, targets = get_batch(
            dataset_train, 
            batch_size=cfg.data.batch_size,
            context_length=cfg.model.context_length,
            device=device
        )
        lr = lr_cosine_schedule(iteration, cfg.train.max_lr, cfg.train.min_lr, cfg.train.warmup_iters, cfg.train.max_iters)
        for group in optimizer.param_groups:
            group['lr'] = lr

        outputs = model(inputs)

        loss = cross_entropy(outputs, targets)
        loss.backward()

        grad_norm = gradient_clipping(model.parameters(), cfg.train.max_l2_norm)
        optimizer.step()
        optimizer.zero_grad()

        if iteration % cfg.train.log_interval == 0:
            tqdm.write(f"Iter:{iteration} Train Loss: {loss.item():.4f} Time: {time.time()-start_time}")
            wandb.log(
                {'train_loss': loss.item(), 'lr':lr, 'grad_norm':grad_norm},
                step = iteration
            )

        if iteration % cfg.train.eval_interval == 0:
            model.eval()
            losses = []
            for eval_iter in tqdm(range(cfg.train.eval_iters), desc="Evaluating", leave=False):
                inputs, targets = get_batch(
                    dataset_test, 
                    batch_size=cfg.data.batch_size,
                    context_length=cfg.model.context_length,
                    device=device
                )
                outputs = model(inputs)
                loss = cross_entropy(outputs, targets)
                losses.append(loss.item())

            tqdm.write(f"Iter:{iteration} Eval Loss: {loss.item():.4f} Time: {time.time()-start_time}")
            wandb.log(
                {'eval_loss': np.mean(losses)},
                step = iteration
            )
            model.train()

        if iteration % cfg.train.checkpoint_interval == 0:
            checkpoint_path = output_dir / f'ckpt_{iteration}.pt'
            tqdm.write(f"Saving checkpoint to {checkpoint_path}")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)

def generated(cfg):
    device = torch.device('cuda:0')
    print("==" * 10, "Load Model", "==" * 10)
    model = TransformerLM.from_file(cfg.model, cfg.data.ckpt_path).to(device)
    print("==" * 10,  "Load Tokenizer", "==" * 10)
    tokenizer = Tokenizer.from_files(cfg.data.vocab_path, cfg.data.merges_path, special_tokens='<|endoftext|>')
    user_input = "Once upon a time there was a little boy named Ben"
    print(f"user input: {user_input}")
    inputs = tokenizer.encode(user_input)
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    outputs = model.generate(inputs, max_new_tokens=256, temperature=0.95, top_k=0.9)[0].tolist()
    print(f"output token length: {len(outputs)}")
    outputs = tokenizer.decode(outputs)
    print(f"model output: {outputs}")
    

if __name__ == "__main__":
    cfg = load_cfg()
    # train(cfg)
    generated(cfg)
    
