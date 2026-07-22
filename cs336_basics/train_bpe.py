import os
import regex as re

from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def init_vocab(special_tokens: list[str]):
    init_vacab = {}
    for i in range(256):
        init_vacab[i] = bytes([i])

    for i, token_str in enumerate(special_tokens, start=256):
        token_bytes = token_str.encode("utf-8")
        init_vacab[i] = token_bytes
    return init_vacab

def create_chunks(input_path: str):
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            yield chunk


def prepare_data(chunks:list[str], special_tokens:list[str]):
    worker = partial(get_pretoken_counter, special_tokens=special_tokens)
    with Pool(processes=4) as pool:
        pretoken_counter_list = pool.map(worker, chunks)

    pretoken_counter = Counter()
    for counter in pretoken_counter_list:
        pretoken_counter.update(counter)

    pairs_counter = Counter()
    for token, count in pretoken_counter.items():
        for start, end in zip(token[:-1], token[1:]):
            pairs_counter[(start, end)] += count

    return pairs_counter, pretoken_counter


def get_pretoken_counter(chunk: str, special_tokens: list[str]):
    local_pretoken_counter = Counter()
    clean_chunk_list = re.split('|'.join([re.escape(token) for token in special_tokens]), chunk)

    for chunk in clean_chunk_list:
        pretoken_list = re.findall(PAT, chunk)
        for pretoken in pretoken_list:
            pretoken_bytes = pretoken.encode('utf-8')
            token = tuple(bytes([byte_value]) for byte_value in pretoken_bytes)
            local_pretoken_counter[token] += 1

    return local_pretoken_counter


def merge(vacab:dict[int, bytes], merges:list[(bytes, bytes)], pairs_counter:Counter[tuple], pretoken_counter:Counter[tuple]):
    selected_pair = max(pairs_counter, key=lambda pair:(pairs_counter.get(pair), pair))

    selected_token = []
    for token in pretoken_counter:
        for start, end in zip(token[:-1], token[1:]):
            if (start, end) == selected_pair:
                selected_token.append(token)
                break
    
    for token in selected_token:
        count = pretoken_counter.get(token)
        for i in range(len(token)-1):
            pairs_counter[(token[i], token[i+1])] -= count
    
    for token in selected_token:
        count = pretoken_counter.get(token)
        new_token = []
        i=0
        while i < len(token):
            if (i + 1 < len(token) and (token[i], token[i + 1]) == selected_pair):
                new_token.append(token[i] + token[i + 1])
                i += 2
            else:
                new_token.append(token[i])
                i += 1

        for i in range(len(new_token)-1):
            pairs_counter[(new_token[i], new_token[i+1])] += count
        
        pretoken_counter.pop(token, None)
        pretoken_counter[tuple(new_token)] += count

    vacab[len(vacab)] = (selected_pair[0] + selected_pair[1])
    merges.append(selected_pair)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:


    merges = []
    vocab = init_vocab(special_tokens)
    chunks = create_chunks(input_path=input_path)
    pairs_counter, pretoken_counter = prepare_data(chunks = chunks, special_tokens=special_tokens)

    while len(vocab) < vocab_size:
        merge(vocab, merges, pairs_counter, pretoken_counter)

    return vocab, merges
    
