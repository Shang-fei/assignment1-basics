from collections.abc import Iterable
from typing import Iterator

import regex as re
import pickle
class Tokenizer:

    def __init__(self, vocab:dict[int, bytes], merges:list[tuple[bytes, bytes]], special_tokens:list[str]):
        self.vocab = vocab
        self.bytes2int = {value: key for key ,value in self.vocab.items()}

        self.merges = merges
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}

        self.special_tokens = special_tokens

        if special_tokens is not None:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)

    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        
        special_tokens = sorted(special_tokens, key=len, reverse=True)
        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        ids:list[int] = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        if self.special_tokens is not None:
            pattern = "(" +  '|'.join([re.escape(token) for token in self.special_tokens])  + ")"
            text_list = re.split(pattern, text)
        else:
            text_list = [text]

        for text in text_list:
            if self.special_tokens is not None:
                token_bytes = text.encode('utf-8')
                if (token_idx:=self.bytes2int.get(token_bytes, None)) is not None:
                    ids.append(token_idx)
                    continue
            
            for pretoken in re.findall(PAT, text):
                token_bytes = [bytes([c]) for c in pretoken.encode('utf-8')]

                while len(token_bytes) >= 2:
                    merge_pair = None
                    merge_rank = float('inf')
                    for start, end in zip(token_bytes[:-1], token_bytes[1:]):
                        pair = (start, end)
                        rank = self.merge_rank.get(pair, None)
                        if rank is not None and rank < merge_rank:
                            merge_rank = rank
                            merge_pair = pair

                    if merge_pair is None:
                        break

                    new_token_bytes = []

                    i = 0
                    while i < len(token_bytes):
                        if (i+1) < len(token_bytes) and (token_bytes[i], token_bytes[i+1]) == merge_pair:
                            new_token_bytes.append(token_bytes[i]+token_bytes[i+1])
                            i += 2

                        else:
                            new_token_bytes.append(token_bytes[i])
                            i += 1
                    token_bytes = new_token_bytes

                for b in token_bytes:
                    ids.append(self.bytes2int[b])

        return ids

        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")
