import os
import heapq
import regex as re

from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class HeapEntry():

    __slots__ = ("count", "pair")

    def __init__(self, count:int, pair:bytes):
        self.count = count
        self.pair = pair

    def __lt__(self, other:"HeapEntry"):
        if self.count != other.count:
            return self.count > other.count

        return self.pair > other.pair

class PairMaxHeap:

    def __init__(self, pairs_counter):
        self.heap = [
            HeapEntry(count, pair) 
            for pair, count in pairs_counter.items()
            if count > 0
        ]
        heapq.heapify(self.heap)

    def push(self, pair, count):
        heapq.heappush(self.heap, HeapEntry(count, pair))

    def pop_max(self, pairs_counter):
        while len(self.heap):
            entry = heapq.heappop(self.heap)
            current_count = pairs_counter.get(entry.pair)

            if current_count == entry.count:
                return entry.pair

        return None

    def maybe_rebuild(self, pairs_counter):
        if len(self.heap) > 3 * len(pairs_counter) + 1 :
            self.heap = [
                HeapEntry(count, pair) 
                for pair, count in pairs_counter.items()
                if count > 0
            ]
            heapq.heapify(self.heap)

    
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
        num_processes = 16
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            yield chunk


def prepare_data(chunks:list[str], special_tokens:list[str]):
    worker = partial(get_pretoken_counter, special_tokens=special_tokens)
    with Pool(processes=8) as pool:
        pretoken_counter_list = pool.map(worker, chunks)

    pretoken_counter = Counter()
    for counter in pretoken_counter_list:
        pretoken_counter.update(counter)

    pairs_counter = Counter()
    for token, count in pretoken_counter.items():
        for start, end in zip(token[:-1], token[1:]):
            pairs_counter[(start, end)] += count

    id2token = dict()
    id2count = defaultdict(int)
    pairs2id = defaultdict(set)

    for id, (token, count) in enumerate(pretoken_counter.items()):
        id2token[id] = token
        id2count[id] = count

        for start, end in zip(token[:-1], token[1:]):
            pairs2id[(start, end)].add(id)

    return pairs_counter, pretoken_counter, id2token, id2count, pairs2id


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


def merge(vacab:dict[int, tuple[bytes]], merges:list[(bytes, bytes)], pairs_counter:Counter[tuple], pretoken_counter:Counter[tuple], 
          heap:PairMaxHeap, id2token:dict[int, tuple[bytes]], id2count:dict[int, int], pairs2id:dict[tuple[bytes], int]):

    selected_pair = heap.pop_max(pairs_counter)

    selected_id = list(pairs2id[selected_pair])
    selected_token = [id2token[id] for id in selected_id]

    changed_pair = Counter() 
    for k, token in enumerate(selected_token):
        idx = selected_id[k]
        count = id2count[idx]
        for i in range(len(token)-1):
            pairs_counter[(token[i], token[i+1])] -= count
            changed_pair[(token[i], token[i+1])] -= count
            pairs2id[(token[i], token[i+1])].discard(idx)

    for k, token in enumerate(selected_token):

        idx = selected_id[k]
        count = id2count[idx]

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
            changed_pair[(new_token[i], new_token[i+1])] += count
            pairs2id[(new_token[i], new_token[i+1])].add(idx)
        
        id2token[idx] = tuple(new_token)

    for pair, delta in changed_pair.items():
        current_count = pairs_counter.get(pair)

        if current_count == 0:
            del pairs_counter[pair]

        elif delta != 0:
            heap.push(pair, current_count)


    heap.maybe_rebuild(pairs_counter)

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
    pairs_counter, pretoken_counter, id2token, id2count, pairs2id = prepare_data(chunks = chunks, special_tokens=special_tokens)

    heap = PairMaxHeap(pairs_counter)
    while len(vocab) < vocab_size:
        merge(vocab, merges, pairs_counter, pretoken_counter, heap, id2token, id2count, pairs2id)

    return vocab, merges
    
