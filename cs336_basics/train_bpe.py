import time
import resource
import pickle

from bpe import train_bpe

def train(input_path:str, vocab_size:int, special_tokens:list[str]):
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    return vocab, merges

if __name__ == "__main__":
    input_path = "/home/shangfei/shang/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    start = time.perf_counter()
    vocab, merges = train(input_path, vocab_size, special_tokens)
    end = time.perf_counter()

    cpu_peak_gb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/ 1024**2)

    print(f"训练用时:{end - start}")
    print(f"CPU 峰值内存: {cpu_peak_gb:.3f} GB")

    with open("../output/tiny_stores_vocab_valid.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("../output/tiny_stores_merges_valid.pkl", "wb") as f:
        pickle.dump(merges, f)
    
    longest_token_id = max(vocab, key = lambda idx : len(vocab.get(idx)))
    longest_token_bytes = vocab.get(longest_token_id)
    longest_token_str = longest_token_bytes.decode('utf-8')
    print(longest_token_str)