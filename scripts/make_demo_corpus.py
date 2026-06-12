from pathlib import Path

sample = Path("data/sample_corpus.txt").read_text(encoding="utf-8")
out = Path("data/processed/corpus.txt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text((sample + "\n") * 2000, encoding="utf-8")
print("wrote demo corpus to", out)
