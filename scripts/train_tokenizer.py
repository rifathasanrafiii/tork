import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab-size", type=int, default=2000)
    args = parser.parse_args()

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    )
    tokenizer.train([args.input], trainer)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.out)
    print(f"saved tokenizer to {args.out}")


if __name__ == "__main__":
    main()
