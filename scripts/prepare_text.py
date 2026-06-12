import argparse
from pathlib import Path


def clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = " ".join(line.strip().split())
        if line:
            lines.append(line)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    texts = []
    for path in sorted(input_dir.glob("*.txt")):
        texts.append(clean_text(path.read_text(encoding="utf-8", errors="ignore")))
    if not texts:
        raise SystemExit("No .txt files found")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(texts), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
