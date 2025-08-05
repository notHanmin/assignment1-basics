#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/minko/cs336/assignment1-basics/tests')

from tests.adapters import get_tokenizer
from tests.common import gpt2_bytes_to_unicode
import json

# Test exactly what the test does
def get_tokenizer_from_vocab_merges_path(
    vocab_path,
    merges_path,
    special_tokens=None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return get_tokenizer(vocab, merges, special_tokens)

# Load using test method  
tokenizer = get_tokenizer_from_vocab_merges_path(
    "tests/fixtures/gpt2_vocab.json",
    "tests/fixtures/gpt2_merges.txt", 
    special_tokens=["<|endoftext|>"]
)

test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
print(f"Original string: {test_string}")

encoded_ids = tokenizer.encode(test_string)
print(f"Encoded IDs: {encoded_ids}")

# Check what each individual token decodes to
individual_tokens = [tokenizer.decode([x]) for x in encoded_ids]
print(f"Individual tokens: {individual_tokens}")

print(f"Count of '<|endoftext|>' in individual tokens: {individual_tokens.count('<|endoftext|>')}")

# Find all positions where we have special tokens
for i, token in enumerate(individual_tokens):
    if '<|endoftext|>' in str(token):
        print(f"Position {i}: ID {encoded_ids[i]}, decoded as: '{token}' (len={len(token)})")

# Check vocab for special token
special_token_bytes = "<|endoftext|>".encode("utf-8")
special_token_ids = []
for token_id, token_bytes in tokenizer.vocab.items():
    if token_bytes == special_token_bytes:
        special_token_ids.append(token_id)

print(f"Special token IDs in vocab: {special_token_ids}")

# Full decode
decoded_string = tokenizer.decode(encoded_ids)
print(f"Full decode: {decoded_string}")
print(f"Full decode matches original: {test_string == decoded_string}")
