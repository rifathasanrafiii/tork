from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer

from torkai.model import GPT, GPTConfig

app = FastAPI(title="Tork AI")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 80


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(request: GenerateRequest):
    checkpoint_path = Path("checkpoints/tork_tiny.pt")
    tokenizer_path = Path("artifacts/tokenizer.json")
    if not checkpoint_path.exists() or not tokenizer_path.exists():
        return {"error": "Train the model first using the README commands."}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    config = GPTConfig(**checkpoint["config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ids = tokenizer.encode(request.prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, request.max_new_tokens)
    return {"text": tokenizer.decode(y[0].tolist())}
