import argparse
import json
from adapters import Tokenizer, run_train_bpe
from common import gpt2_bytes_to_unicode

def main():

    corpus_filepath = '/home/minko/cs336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt'
    vocab_filepath = '/home/minko/cs336/assignment1-basics/tests/fixtures/gpt2_vocab.json'
    merges_filepath = '/home/minko/cs336/assignment1-basics/tests/fixtures/gpt2_merges.txt'

    parser = argparse.ArgumentParser(description="Train a tokenizer.")
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size.')

    args = parser.parse_args()

    special_tokens = ['<|endoftext|>']

    #vocab, merges in bytes
    vocab, merges = run_train_bpe(corpus_filepath, args.vocab_size, special_tokens)

    byte_encoder = gpt2_bytes_to_unicode()
    unicode_encoder = {v: k for k, v in byte_encoder.items()}
    vocab_to_save = {
        "".join([byte_encoder[b] for b in token_bytes]): token_id
        for token_id, token_bytes in vocab.items()
    }

    new_vocab_path = '/home/minko/cs336/assignment1-basics/tests/fixtures/new_vocab.json'
    new_merges_path = '/home/minko/cs336/assignment1-basics/tests/fixtures/new_merges.txt'

    print(f"Saving vocabulary to {new_vocab_path}...")
    with open(new_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_to_save, f, ensure_ascii=False, indent=2)

    with open(new_merges_path, 'w', encoding='utf-8') as f:
        f.write("#version: 0.2\n")
        for b1, b2 in merges:
            s1 = "".join([byte_encoder[b] for b in b1])
            s2 = "".join([byte_encoder[b] for b in b2])
            f.write(f"{s1} {s2}\n")

if __name__ == "__main__":
    main()
