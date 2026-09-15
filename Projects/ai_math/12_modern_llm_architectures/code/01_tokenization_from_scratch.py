"""
Modern Tokenization from Scratch: Byte-Pair Encoding (BPE) Verification Suite
Module 12: Modern LLM Architectures - Chapter 01

This script verifies:
1. Exact numerical reproduction of Part 5 'AI by Hand' calculations:
   - 4-step BPE merge extraction on toy corpus
   - Exact tokenization of unseen word 'lowest' -> ['lo', 'w', 'est_']
2. Complete Byte-Level BPE Tokenizer:
   - UTF-8 byte stream processing (0-255 base alphabet)
   - Pair frequency calculation and greedy merge table construction
   - Exact round-trip decode(encode(text)) == text across English, multilingual, and emojis.
"""

from collections import defaultdict


# =====================================================================
# 1. Part 5 'AI by Hand' Numerical Verification
# =====================================================================

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = f"{pair[0]} {pair[1]}"
    replacement = f"{pair[0]}{pair[1]}"
    for word in v_in:
        w_out = word.replace(bigram, replacement)
        v_out[w_out] = v_in[word]
    return v_out


def verify_part5_hand_calculation():
    print("=" * 70)
    print("1. VERIFYING PART 5 'AI BY HAND' BPE TRAINING CALCULATIONS")
    print("=" * 70)

    # Initial corpus with end-of-word marker '_'
    vocab = {
        "l o w _": 5,
        "l o w e r _": 2,
        "n e w e s t _": 6,
        "w i d e s t _": 3,
    }

    expected_merges = [
        (("e", "s"), "es", 9),
        (("es", "t"), "est", 9),
        (("est", "_"), "est_", 9),
        (("l", "o"), "lo", 7),
    ]

    merges = []
    for step in range(4):
        pairs = get_stats(vocab)
        # Select best pair with tie-breaking: highest frequency, then alphabetical
        best_pair = max(pairs.keys(), key=lambda p: (pairs[p], -ord(p[0][0])))
        freq = pairs[best_pair]

        exp_pair, exp_token, exp_freq = expected_merges[step]
        print(f"Step {step + 1}: Selected {best_pair} with freq {freq} (Expected: {exp_pair} freq {exp_freq})")

        assert best_pair == exp_pair, f"Expected {exp_pair}, got {best_pair}"
        assert freq == exp_freq, f"Expected freq {exp_freq}, got {freq}"

        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)

    print("Updated Corpus Vocabulary:")
    for w, f in vocab.items():
        print(f"  '{w}': {f}")

    # Test encoding on unseen word 'lowest'
    word_tokens = "l o w e s t _".split()
    for pair in merges:
        i = 0
        new_tokens = []
        while i < len(word_tokens):
            if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == pair:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(word_tokens[i])
                i += 1
        word_tokens = new_tokens

    print(f"Tokenization of unseen word 'lowest': {word_tokens} (Expected: ['lo', 'w', 'est_'])")
    assert word_tokens == ["lo", "w", "est_"], f"Expected ['lo', 'w', 'est_'], got {word_tokens}"
    print(">> SUCCESS: Hand-derived BPE merges and unseen tokenization perfectly verified!\n")


# =====================================================================
# 2. Complete Byte-Level BPE Tokenizer from Scratch
# =====================================================================

class ByteLevelBPETokenizer:
    """
    Byte-Level BPE Tokenizer supporting arbitrary Unicode text without OOV.
    """
    def __init__(self):
        # Base vocabulary: 256 individual bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.inverse_vocab = {bytes([i]): i for i in range(256)}
        self.merges = {}  # (p0, p1) -> new_id

    def train(self, text: str, vocab_size: int):
        assert vocab_size >= 256, "Vocab size must be at least 256 for byte alphabet"
        num_merges = vocab_size - 256

        # Convert text into sequence of byte integers
        tokens = list(text.encode("utf-8"))

        for i in range(num_merges):
            # Count bigram pairs
            stats = defaultdict(int)
            for p0, p1 in zip(tokens[:-1], tokens[1:]):
                stats[(p0, p1)] += 1

            if not stats:
                break

            # Find most frequent pair
            best_pair = max(stats.keys(), key=lambda p: stats[p])
            new_id = 256 + i

            # Register new merged token
            self.merges[best_pair] = new_id
            merged_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = merged_bytes
            self.inverse_vocab[merged_bytes] = new_id

            # Replace pair in tokens list
            new_tokens = []
            idx = 0
            while idx < len(tokens):
                if idx < len(tokens) - 1 and (tokens[idx], tokens[idx + 1]) == best_pair:
                    new_tokens.append(new_id)
                    idx += 2
                else:
                    new_tokens.append(tokens[idx])
                    idx += 1
            tokens = new_tokens

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            # Find candidate pairs present in our merges
            stats = {}
            for idx, pair in enumerate(zip(tokens[:-1], tokens[1:])):
                if pair in self.merges:
                    # Priority is given to merges created earlier (lower new_id)
                    stats[pair] = self.merges[pair]

            if not stats:
                break  # No more mergeable pairs

            # Select pair with lowest merge index
            pair_to_merge = min(stats.keys(), key=lambda p: stats[p])
            new_id = self.merges[pair_to_merge]

            new_tokens = []
            idx = 0
            while idx < len(tokens):
                if idx < len(tokens) - 1 and (tokens[idx], tokens[idx + 1]) == pair_to_merge:
                    new_tokens.append(new_id)
                    idx += 2
                else:
                    new_tokens.append(tokens[idx])
                    idx += 1
            tokens = new_tokens

        return tokens

    def decode(self, token_ids: list[int]) -> str:
        byte_stream = b"".join(self.vocab[tid] for tid in token_ids)
        return byte_stream.decode("utf-8", errors="replace")


def verify_byte_level_bpe():
    print("=" * 70)
    print("2. VERIFYING BYTE-LEVEL BPE TOKENIZER ON MULTILINGUAL TEXT & CODE")
    print("=" * 70)

    corpus = (
        "The quick brown fox jumps over the lazy dog. "
        "Large language models and Transformers are neural networks. "
        "def forward(self, x):\n    return self.linear(x)\n"
        "Machine learning, deep learning, and reinforcement learning. "
        "Mathematics: e^{i\\pi} + 1 = 0. Emojis: 🚀 🤖 🧠."
    )

    tokenizer = ByteLevelBPETokenizer()
    target_vocab_size = 300  # 256 base bytes + 44 learned merges
    tokenizer.train(corpus, vocab_size=target_vocab_size)

    print(f"Trained BBPE Tokenizer with {len(tokenizer.merges)} learned merge rules.")
    assert len(tokenizer.vocab) == target_vocab_size

    # Test round-trip exact reconstruction
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "def attention(Q, K, V): return softmax(Q @ K.T) @ V",
        "Bonjour le monde! 🚀 DeepSeek & LLaMA-3.",
        "Unseen novel token sequence with emojis: 🌟✨🔥",
    ]

    for s in test_sentences:
        encoded = tokenizer.encode(s)
        decoded = tokenizer.decode(encoded)
        compression = len(s.encode('utf-8')) / len(encoded)
        print(f"Original:   '{s}'")
        print(f"Encoded:    {encoded[:8]}... (Total {len(encoded)} tokens, Compression {compression:.2f}x)")
        print(f"Decoded:    '{decoded}'")
        assert decoded == s, f"Round-trip decoding mismatch: '{decoded}' != '{s}'"
        print("  [PASSED] Exact round-trip match verified.")

    print("\n>> SUCCESS: Byte-Level BPE tokenizer successfully trained and passed all round-trip tests!\n")


# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    verify_part5_hand_calculation()
    verify_byte_level_bpe()
    print("=" * 70)
    print("ALL MODULE 12 CHAPTER 01 (TOKENIZATION FROM SCRATCH) VERIFICATIONS PASSED!")
    print("=" * 70)
