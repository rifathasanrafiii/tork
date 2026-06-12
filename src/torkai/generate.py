import argparse

import torch
from tokenizers import Tokenizer

from torkai.model import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ids = tokenizer.encode(args.prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
