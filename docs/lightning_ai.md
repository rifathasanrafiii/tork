# Train Tork AI on Lightning AI Studio

## 1. Open a new Studio

Go to Lightning AI, create a Studio, and select a GPU machine if available.

## 2. Open terminal

Inside the Studio, open Terminal.

## 3. Clone the repo

```bash
git clone https://github.com/rifathasanrafiii/tork.git
cd tork
```

## 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 5. Prepare training data

Put your `.txt` files inside `data/raw/`.

Then run:

```bash
python scripts/prepare_text.py --input-dir data/raw --out data/processed/corpus.txt
```

If you only want to test quickly, run:

```bash
mkdir -p data/processed
cp data/sample_corpus.txt data/processed/corpus.txt
```

## 6. Train tokenizer

```bash
python scripts/train_tokenizer.py --input data/processed/corpus.txt --out artifacts/tokenizer.json --vocab-size 8000
```

## 7. Train 10M model

```bash
python -m torkai.train --config configs/10m.json
```

## 8. Generate text

```bash
python -m torkai.generate --checkpoint checkpoints/tork_10m.pt --tokenizer artifacts/tokenizer.json --prompt "Ami ekta AI banate chai" --max-new-tokens 100
```

## 9. Run API

```bash
uvicorn torkai.api:app --host 0.0.0.0 --port 8000
```

## Notes

- The sample corpus is only for testing. Real output needs much larger and cleaner text data.
- If CUDA is available, the training script automatically uses GPU.
- If you get out-of-memory errors, reduce `batch_size` in `configs/10m.json` from 16 to 8 or 4.
