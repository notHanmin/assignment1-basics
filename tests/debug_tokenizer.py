import os
from typing import BinaryIO
from multiprocessing import Pool
import regex as re
from collections import Counter
import json
import tiktoken

'''
data_path = "data/TinyStoriesV2-GPT4-valid.txt"
with open(data_path, "rb") as f:
    f.seek(0)
    chunk_bytes = f.read(1000   )
    test_string = chunk_bytes.decode("utf-8", errors="ignore")
'''

def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

def apply_bpe(part_bytes: bytes) -> list[int]:
    if not part_bytes:
        return []

    parts = [bytes([b]) for b in part_bytes]

    while True:
        best_pair_to_merge = None
        min_rank = float('inf')

        for i in range(len(parts) - 1):
            pair = (parts[i], parts[i+1])
            if pair in merge_ranks and merge_ranks[pair] < min_rank:
                min_rank = merge_ranks[pair]
                best_pair_to_merge = pair
        
        if best_pair_to_merge is None:
            break

        new_parts = []
        i = 0
        while i < len(parts):
            if i < len(parts) - 1 and (parts[i], parts[i+1]) == best_pair_to_merge:
                new_parts.append(parts[i] + parts[i+1])
                i += 2
            else:
                new_parts.append(parts[i])
                i += 1
        parts = new_parts
    
    # Convert the final byte parts to token IDs
    return [byte_to_id[p] for p in parts]

def decode(ids: list[int]) -> str:
    output = b"".join(vocab[id] for id in ids)
    return output.decode("utf-8", errors="replace")

test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"]

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pre_tokenize_regex = re.compile(PAT)
special_tokens_pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
special_tokens_regex = re.compile(special_tokens_pattern)

corpus_path = "/home/minko/cs336/assignment1-basics/tests/fixtures/address.txt"

with open(corpus_path) as f:
    corpus_contents = f.read()

tokenized_ids = []
chunks = special_tokens_regex.split(corpus_contents)

print(f"Text: {test_string}")
print(f"Chunks: {chunks}")
print(f"Special tokens: {special_tokens}")

vocab_filepath = "/home/minko/cs336/assignment1-basics/tests/fixtures/gpt2_vocab.json"
merges_filepath = "/home/minko/cs336/assignment1-basics/tests/fixtures/gpt2_merges.txt"

with open(vocab_filepath, 'r', encoding = 'utf-8') as f:
            gpt2_vocab = json.load(f)
with open(merges_filepath, 'r', encoding = 'utf-8') as f:
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
byte_to_id = {v: k for k, v in vocab.items()}
merge_ranks = {pair: i for i, pair in enumerate(merges)}

for chunk in chunks:
    if not chunk:
        continue
    if chunk in special_tokens:
        tokenized_ids.append(byte_to_id[chunk.encode("utf-8")])
    else:
        pre_tokenized_parts = pre_tokenize_regex.findall(chunk)

        for part in pre_tokenized_parts:
            part_bytes = part.encode("utf-8")
            ids = apply_bpe(part_bytes)
            tokenized_ids.extend(ids)

print(f"{tokenized_ids}")

tokenized_string = [decode([x]) for x in tokenized_ids]
print(f"{tokenized_string}")
decoded_string = decode(tokenized_ids)
print(f"{decoded_string}")

reference_tokenizer = tiktoken.get_encoding("gpt2")

reference_ids = reference_tokenizer.encode(corpus_contents)
print(tokenized_ids == reference_ids)