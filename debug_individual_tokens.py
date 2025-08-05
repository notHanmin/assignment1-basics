#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/minko/cs336/assignment1-basics/tests')

from tests.adapters import get_tokenizer
from tests.common import gpt2_bytes_to_unicode
import json

# Load the tokenizer the same way the test does
def get_tokenizer_from_vocab_merges_path(vocab_path, merges_path, special_tokens=None):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        gpt2_vocab = json.load(f)
    
    with open(merges_path, 'r', encoding='utf-8') as f:
        bpe_merges_from_file = f.read().split('\n')[1:-1]
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    
    vocab = {
        index: bytes([gpt2_byte_decoder[char] for char in token])
        for token, index in gpt2_vocab.items()
    }
    
    merges = []
    for merge_rule in bpe_merges_from_file:
        p1, p2 = merge_rule.split()
        b1 = bytes([gpt2_byte_decoder[char] for char in p1])
        b2 = bytes([gpt2_byte_decoder[char] for char in p2])
        merges.append((b1, b2))
    
    if special_tokens:
        for token in special_tokens:
            byte_encoded = token.encode("utf-8")
            if byte_encoded not in vocab.values():
                vocab[len(vocab)] = byte_encoded
    
    return get_tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

# Test the exact same scenario
tokenizer = get_tokenizer_from_vocab_merges_path(
    vocab_path="tests/fixtures/gpt2_vocab.json",
    merges_path="tests/fixtures/gpt2_merges.txt", 
    special_tokens=["<|endoftext|>"]
)

test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
print(f"Original string: {test_string}")

encoded_ids = tokenizer.encode(test_string)
print(f"Encoded IDs: {encoded_ids}")

# Check what each individual token decodes to
tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
print(f"Individual tokens decoded: {tokenized_string}")

# Check for special tokens specifically
special_token_positions = []
for i, token in enumerate(tokenized_string):
    if token == "<|endoftext|>":
        special_token_positions.append(i)
        
print(f"Special token positions: {special_token_positions}")
print(f"Count of '<|endoftext|>' in tokenized_string: {tokenized_string.count('<|endoftext|>')}")

# Let's also check the vocab for the special token
special_token_bytes = "<|endoftext|>".encode("utf-8")
special_token_id = None
for token_id, token_bytes in tokenizer.vocab.items():
    if token_bytes == special_token_bytes:
        special_token_id = token_id
        break

if special_token_id is not None:
    print(f"Special token ID: {special_token_id}")
    print(f"Special token bytes: {tokenizer.vocab[special_token_id]}")
    print(f"Decoding special token ID: '{tokenizer.decode([special_token_id])}'")
else:
    print("Special token not found in vocab!")

# Full decode
decoded_string = tokenizer.decode(encoded_ids)
print(f"Full decode: {decoded_string}")
print(f"Full decode matches original: {test_string == decoded_string}")
