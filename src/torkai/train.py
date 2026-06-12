import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torkai.model import GPT, GPTConfig


class TextDataset(Dataset):
    def __init__(self, text_path, tokenizer_path, block_size):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        text = Path(text_path).read_text(encoding="utf-8")
        ids = self.tokenizer.encode(text).ids
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size
        if len(self.data) <= block_size + 1:
            raise ValueError("Not enough text data for this block_size")

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny.json")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(cfg["tokenizer_path"])

    model_config = GPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg.get("dropout", 0.1),
    )
    model = GPT(model_config).to(device)
    print("device", device)
    print("parameters", model.parameter_count())

    dataset = TextDataset(cfg["train_text"], cfg["tokenizer_path"], cfg["block_size"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(cfg["epochs"]):
        for x, y in tqdm(loader, desc=f"epoch {epoch + 1}"):
            x = x.to(device)
            y = y.to(device)
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 10 == 0:
                print("step", step, "loss", round(loss.item(), 4))
            if step >= cfg.get("max_steps", 1000):
                break
        ckpt = Path(cfg["checkpoint_dir"]) / cfg["checkpoint_name"]
        torch.save({"model": model.state_dict(), "config": model_config.__dict__, "step": step}, ckpt)
        print("saved", ckpt)
        if step >= cfg.get("max_steps", 1000):
            break


if __name__ == "__main__":
    main()
