# Tork AI

Tork AI is a from-scratch mini GPT-style language model project. It is designed as a realistic first step toward your own Bangla-English AI model.

This is **not** an API wrapper. The model architecture, tokenizer training, data pipeline, training loop, checkpointing, generation script, FastAPI server, and Streamlit chat UI are included in this repo.

## What this repo gives you

- GPT-style Transformer model written in PyTorch
- Trainable tokenizer using Hugging Face `tokenizers`
- Dataset preparation for plain `.txt` files
- Training script with checkpoint saving
- Text generation / chat-style script
- FastAPI inference API
- Streamlit web chat UI
- GitHub Actions smoke test
- Dockerfile

## Realistic first target

Start with a **tiny 1M-10M parameter model** to confirm the pipeline works. Then move to **10M-50M parameters** on a free GPU platform such as Lightning AI Studio, Kaggle, or Colab.

A 5B+ model requires serious GPU compute and is not realistic on a normal laptop or standard GitHub runner.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Train a tokenizer from the sample corpus:

```bash
python scripts/train_tokenizer.py --input data/sample_corpus.txt --out artifacts/tokenizer.json --vocab-size 2000
```

Train the tiny model:

```bash
python -m torkai.train --config configs/tiny.json
```

Generate text:

```bash
python -m torkai.generate --checkpoint checkpoints/tork_tiny.pt --tokenizer artifacts/tokenizer.json --prompt "Ami ekta AI banate chai" --max-new-tokens 80
```

Run API:

```bash
uvicorn torkai.api:app --reload
```

Run UI:

```bash
streamlit run app/streamlit_app.py
```

## Add your own data

Put text files inside `data/raw/`, then run:

```bash
python scripts/prepare_text.py --input-dir data/raw --out data/processed/corpus.txt
python scripts/train_tokenizer.py --input data/processed/corpus.txt --out artifacts/tokenizer.json --vocab-size 8000
```

Then update `configs/tiny.json` to point to `data/processed/corpus.txt`.

## Suggested growth path

1. Smoke test: tiny config on CPU
2. Train 10M model on free GPU
3. Add more Bangla-English data
4. Upload checkpoint/tokenizer to Hugging Face
5. Deploy FastAPI + Streamlit/Next.js UI

## Repo status

Initial scaffold created by ChatGPT for Rifat / Raaf.
