import os
from typing import BinaryIO
from multiprocessing import Pool
import regex as re
from collections import Counter
import json

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(args):
    file_path, start, end, special_tokens = args

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_str = chunk_bytes.decode("utf-8", errors="ignore")

    delimiter = "|".join(re.escape(token) for token in special_tokens)
    subchunks_str = re.split(delimiter, chunk_str)

    chunk_counter = Counter()
    regex_compiled = re.compile(PAT)
    for subchunk_str in subchunks_str:
        matches = regex_compiled.finditer(subchunk_str)
        chunk_counter.update(match.group(0) for match in matches)

    return chunk_counter

def parallel_pretokenize(
        file_path: str,
        num_processes: int,
        special_tokens: list[str]
) -> Counter:
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))

    chunk_args = [
        (file_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(processes=num_processes) as pool:
        chunk_counters = pool.map(pretokenize, chunk_args)

    total_counter = Counter()
    for counter in chunk_counters:
        total_counter.update(counter)

    return total_counter

## Usage
data_path = "data/TinyStoriesV2-GPT4-valid.txt"
num_processes = os.cpu_count()
if num_processes is None:
    num_processes = 1

special_tokens_list = ["<|endoftext|>"]
pretoken_counts = parallel_pretokenize(data_path, num_processes, special_tokens_list)

count = pretoken_counts.get("<|endoftext|>", 0)
print(f"Occurrences of ' and': {count}")

top_10_occurrences = pretoken_counts.most_common(10)

print("--- Top 10 Occurrences ---")
for item, count in top_10_occurrences:
    item_encoded = item.encode("utf-8")
    print(f"{item_encoded}: {count}")

with open(data_path, "rb") as f:
    f.seek(0)
    chunk_bytes = f.read(1000   )
    test_string = chunk_bytes.decode("utf-8", errors="ignore")

#test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
special_tokens=["<|endoftext|>"]
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pre_tokenize_regex = re.compile(PAT)
special_tokens_pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
special_tokens_regex = re.compile(special_tokens_pattern)

tokenized_ids = []
chunks = special_tokens_regex.split(test_string)

print(f"Text: {test_string}")
print(f"Chunks: {chunks}")
print(f"Special tokens: {special_tokens}")
vocab_filepath = "/home/minko/cs336/assignment1-basics/tests/fixtures/gpt2_vocab.json"
with open(vocab_filepath, 'r', encoding = 'utf-8') as f:
            gpt2_vocab = json.load(f)
