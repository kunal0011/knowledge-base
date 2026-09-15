# Chapter 01: Modern Tokenization from Scratch: BPE, Byte-Level BPE & Tiktoken

---

## 1. Intuition & 101 Motivation

Before a Large Language Model can process text, the text must be translated from a stream of human characters into an array of discrete numerical identifiers (**tokens**):
$$\text{"Hello, world!"} \longrightarrow [15496, 11, 995, 0]$$

Choosing how to slice text into tokens presents a fundamental engineering trade-off:
1. **Character-Level Tokenization:** Vocabulary is tiny (256 ASCII characters or a few thousand Unicode symbols). However, text sequences become extremely long. Because standard self-attention compute scales quadratically with sequence length ($O(T^2)$), character-level models are computationally expensive and struggle to capture long-range semantic dependencies.
2. **Word-Level Tokenization:** Sequences are short, but the vocabulary explodes into millions of words. Furthermore, it suffers from the fatal **Out-of-Vocabulary (OOV)** problem: any rare inflection, typo, or new word is replaced by an uninformative `<unk>` token.
3. **Subword Tokenization (The Golden Mean):** Frequent words remain complete tokens (e.g., `"apple"`), while rare or compound words are decomposed into common subword chunks (e.g., `"un"` + `"predict"` + `"able"`).

Today, virtually all premier foundation models—from GPT-4 and LLaMA 3 to Mistral, Gemma, and DeepSeek—rely on **Byte-Level Byte-Pair Encoding (BBPE)**.

In this chapter, inspired by Sebastian Raschka's *Build a Large Language Model (From Scratch)*, we build a complete BPE tokenizer from first principles: from raw byte sequences and statistical pair merging to regex boundary preservation and inference encoding.

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Classic Byte-Pair Encoding (BPE) Algorithm

Byte-Pair Encoding (Sennrich et al., 2016; originally a data compression algorithm by Philip Gage, 1994) iteratively builds a subword vocabulary by greedily merging the most frequent adjacent symbol pairs.

#### The Training Process:
1. **Vocabulary Initialization:**
   Initialize the base vocabulary $V_0$ with all atomic characters present in training corpus $\mathcal{C}$ plus an end-of-word marker `</w>`:
   $$V_0 = \Sigma \cup \{ \text{</w>} \}$$
2. **Tokenized Corpus State:**
   Every word $w \in \mathcal{C}$ is split into individual characters:
   $$w = (c_1, c_2, \dots, c_L, \text{</w>})$$
3. **Frequency Calculation:**
   For each adjacent pair of symbols $(u, v)$, compute its total frequency across the entire corpus:
   $$\operatorname{Freq}(u, v) = \sum_{w \in \mathcal{C}} \operatorname{Count}(w) \cdot \operatorname{Occurrences}((u, v) \text{ in } w)$$
4. **Greedy Merge Selection:**
   Identify the pair with maximum frequency:
   $$(u^*, v^*) = \arg\max_{(u, v)} \operatorname{Freq}(u, v)$$
5. **Vocabulary Augmentation & Corpus Rewrite:**
   Create a new composite symbol $uv = u^* \circ v^*$:
   $$V_{k+1} = V_k \cup \{ uv \}$$
   Record the merge rule $(u^*, v^*) \to uv$ in ordered merge table $\mathcal{M}$.
   Replace all occurrences of $(u^*, v^*)$ with $uv$ throughout corpus $\mathcal{C}$.
6. **Termination:**
   Repeat steps 3–5 until $|V| = V_{\text{target}}$ or the maximum frequency drops below a threshold.

---

### 2.2 Byte-Level BPE (BBPE) & Universal Unicode Coverage

Standard BPE breaks down when handling multilingual text, emojis, or non-Latin scripts, as the base character set $\Sigma$ can contain tens of thousands of rare Unicode characters.

**Byte-Level BPE (GPT-2, GPT-4, LLaMA 3)** solves this elegantly:
- The base alphabet is fixed to the **256 possible values of a single byte** ($0x00$ through $0xFF$).
- Any text is first encoded into raw bytes using the standard **UTF-8** variable-length encoding:
  - ASCII characters $\to 1$ byte ($0x00 - 0x7F$).
  - Latin accents, Greek, Cyrillic, Arabic $\to 2$ bytes.
  - Chinese, Japanese, Korean $\to 3$ bytes.
  - Emojis (e.g., 🚀) $\to 4$ bytes.
- BPE merges are executed directly over byte sequences!

> **Theorem 12.1 (Universal OOV Immunity):**
> Because any valid or invalid text sequence can be decomposed into a sequence of bytes $\in \{0, \dots, 255\}$, a Byte-Level BPE tokenizer **never encounters an unknown token `<unk>`**. Its vocabulary covers 100% of human language, code, and binary data.

---

### 2.3 Regex Splitting (Preserving Semantic Boundaries)

If BPE is run naively on raw text, it will frequently merge across punctuation, spaces, and grammatical boundaries (e.g., merging `"dog."` or `"end,the"` into single tokens).

To prevent this, modern tokenizers (like OpenAI's `tiktoken` and GPT-4) first split text using a **pre-tokenization regular expression** before applying BPE merges.

The canonical GPT-4 pre-tokenization regex pattern is:
```regex
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

#### Anatomical Breakdown:
- `'s|'t|'re|...`: Isolates English contractions so that `"don't"` becomes `["don", "'t"]`.
- ` ?\p{L}+`: Matches sequences of Unicode letters optionally preceded by a space.
- ` ?\p{N}+`: Matches sequences of Unicode digits optionally preceded by a space (keeping numbers separate from text).
- ` ?[^\s\p{L}\p{N}]+`: Matches punctuation and symbols (e.g., `","`, `"!"`, `"$"`).
- `\s+(?!\S)` and `\s+`: Matches consecutive whitespaces without combining with non-space tokens.

By running BPE independently within each regex match, words are prevented from merging with punctuation or trailing spaces!

---

## 3. Geometric & Physical Interpretation

### 3.1 Compression Ratio & Vocabulary Sizing
Tokenization can be viewed through information theory as an adaptive **Lempel-Ziv dictionary compression transform**:
$$\text{Compression Ratio } \rho = \frac{\text{Number of UTF-8 Bytes}}{\text{Number of Tokens Generated}}$$

```
Token Count T
 ^
 |    [ Character Tokenizer (Vocab ~ 256) ]  ---> Max Sequence Length, Quadratic Attention O(T^2)
 |       \
 |        \
 |         * [ LLaMA 2 (Vocab = 32k, Ratio ~ 3.8 bytes/tok) ]
 |            \
 |             * [ LLaMA 3 / GPT-4 (Vocab = 128k, Ratio ~ 4.8 bytes/tok) ]
 |                \
 |                 [ Word-Level Tokenizer (Vocab ~ 1M+) ] ---> Huge Embedding Layer Memory
 |____________________________________________________________________> Vocabulary Size |V|
```
- Increasing vocabulary size from $32\text{k}$ (LLaMA 2) to $128\text{k}$ (LLaMA 3) compresses token sequences by **$15–25\%$**, directly translating to longer effective context windows and lower generation latency!

---

## 4. Real-World Analogy: Stenography & Shorthand

Imagine court stenographers taking notes at 250 words per minute:
- If they wrote out every individual letter (character-level), their hands could never keep up.
- If their keyboard had a separate key for every English word (word-level), the keyboard would be the size of a football stadium.
- Instead, stenography uses **chords**: pressing combinations of keys representing common syllables and word fragments (`"tion"`, `"pre"`, `"ing"`). Frequent words take a single chord; complex words take two chords.
- That is Byte-Pair Encoding: an optimal shorthand vocabulary automatically derived from corpus frequency statistics!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace the complete BPE training process by hand on a concrete toy training corpus.

---

### 5.1 Training Corpus Setup
Consider a toy text corpus with word frequencies:
$$\mathcal{C} = \{ \text{"low": 5}, \quad \text{"lower": 2}, \quad \text{"newest": 6}, \quad \text{"widest": 3} \}$$

Each word is tokenized into individual characters with an explicit end-of-word marker `_`:
- `w_1`: `l o w _` (Frequency: 5)
- `w_2`: `l o w e r _` (Frequency: 2)
- `w_3`: `n e w e s t _` (Frequency: 6)
- `w_4`: `w i d e s t _` (Frequency: 3)

Initial Base Vocabulary:
$$V_0 = \{ \text{'d', 'e', 'i', 'l', 'n', 'o', 'r', 's', 't', 'w', '\_'} \}$$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| `Pair` | `symbol_pair` | Adjacent tuple of current tokens $(c_i, c_{i+1})$ |
| `Count` | `pair_frequency` | Total corpus-weighted count of this pair |
| `Best Pair` | `max_pair` | $\arg\max \text{Count}$ selected for merge |
| `New Token` | `merged_symbol` | Composite symbol added to vocabulary: $c_i \circ c_{i+1}$ |
| $V_k$ | `vocabulary` | Set of known tokens at iteration $k$ |

---

### 5.3 Step-by-Step Hand Iterations

#### Iteration 1:
Let us tally all adjacent pair counts across the corpus:
- `('l', 'o')`: $5 \times 1 + 2 \times 1 = \mathbf{7}$
- `('o', 'w')`: $5 \times 1 + 2 \times 1 = \mathbf{7}$
- `('w', '_')`: $5 \times 1 = \mathbf{5}$
- `('w', 'e')`: in `lower`: $2 \times 1$, in `newest`: $6 \times 1 \implies 2 + 6 = \mathbf{8}$
- `('e', 'r')`: $2 \times 1 = \mathbf{2}$
- `('r', '_')`: $2 \times 1 = \mathbf{2}$
- `('n', 'e')`: $6 \times 1 = \mathbf{6}$
- `('e', 's')`: in `newest`: $6 \times 1$, in `widest`: $3 \times 1 \implies 6 + 3 = \mathbf{9}$
- `('s', 't')`: in `newest`: $6 \times 1$, in `widest`: $3 \times 1 \implies 6 + 3 = \mathbf{9}$
- `('t', '_')`: in `newest`: $6 \times 1$, in `widest`: $3 \times 1 \implies 6 + 3 = \mathbf{9}$
- `('w', 'i')`: $3 \times 1 = \mathbf{3}$
- `('i', 'd')`: $3 \times 1 = \mathbf{3}$
- `('d', 'e')`: $3 \times 1 = \mathbf{3}$

The highest frequency is **$9$**, tied between `('e', 's')`, `('s', 't')`, and `('t', '_')`.
By alphabetical tie-break: **Select `('e', 's')`**.
- **New Token:** `'es'`
- **Corpus Update:**
  - `w_1`: `l o w _` (5)
  - `w_2`: `l o w e r _` (2)
  - `w_3`: `n e w es t _` (6)
  - `w_4`: `w i d es t _` (3)

---

#### Iteration 2:
Re-calculate pair counts on updated corpus:
- `('es', 't')`: $6 + 3 = \mathbf{9}$
- `('t', '_')`: $6 + 3 = \mathbf{9}$
- `('w', 'es')`: $6 \times 1 = \mathbf{6}$
- `('d', 'es')`: $3 \times 1 = \mathbf{3}$
- `('w', 'e')`: in `lower`: $2 \times 1 = \mathbf{2}$ *(Note: 'w e' dropped from 8 to 2 because 'newest' became 'new es t')*
- `('l', 'o')`: $\mathbf{7}$
- `('o', 'w')`: $\mathbf{7}$

Highest count is **$9$** between `('es', 't')` and `('t', '_')`.
Select `('es', 't')`.
- **New Token:** `'est'`
- **Corpus Update:**
  - `w_1`: `l o w _` (5)
  - `w_2`: `l o w e r _` (2)
  - `w_3`: `n e w est _` (6)
  - `w_4`: `w i d est _` (3)

---

#### Iteration 3:
Pair counts:
- `('est', '_')`: in `newest`: 6, in `widest`: $3 \implies \mathbf{9}$
- `('l', 'o')`: $\mathbf{7}$
- `('o', 'w')`: $\mathbf{7}$
- `('w', 'est')`: $\mathbf{6}$
- `('w', '_')`: $\mathbf{5}$

Highest count is **$9$**: `('est', '_')`.
- **New Token:** `'est_'`
- **Corpus Update:**
  - `w_1`: `l o w _` (5)
  - `w_2`: `l o w e r _` (2)
  - `w_3`: `n e w est_` (6)
  - `w_4`: `w i d est_` (3)

---

#### Iteration 4:
Pair counts:
- `('l', 'o')`: $5 + 2 = \mathbf{7}$
- `('o', 'w')`: $5 + 2 = \mathbf{7}$
- `('w', 'est_')`: $\mathbf{6}$
- `('w', '_')`: $\mathbf{5}$

Select `('l', 'o')` (count 7).
- **New Token:** `'lo'`
- **Corpus Update:**
  - `w_1`: `lo w _` (5)
  - `w_2`: `lo w e r _` (2)
  - `w_3`: `n e w est_` (6)
  - `w_4`: `w i d est_` (3)

---

### 5.4 Summary Visual Grid: BPE Training Ledger

| Iteration | Candidate Pairs Evaluated | Top Frequency | Selected Pair $(u, v)$ | Resulting Token | Updated Word Representations |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | Baseline single characters | — | — | Initial Alphabet | `l o w _`, `l o w e r _`, `n e w e s t _`, `w i d e s t _` |
| **1** | `('e', 's')`: 9, `('s', 't')`: 9, `('w', 'e')`: 8 | **$9$** | `('e', 's')` | `'es'` | `n e w es t _`, `w i d es t _` |
| **2** | `('es', 't')`: 9, `('t', '_')`: 9, `('l', 'o')`: 7 | **$9$** | `('es', 't')` | `'est'` | `n e w est _`, `w i d est _` |
| **3** | `('est', '_')`: 9, `('l', 'o')`: 7, `('o', 'w')`: 7 | **$9$** | `('est', '_')` | `'est_'` | `n e w est_`, `w i d est_` |
| **4** | `('l', 'o')`: 7, `('o', 'w')`: 7, `('w', 'est_')`: 6 | **$7$** | `('l', 'o')` | `'lo'` | `lo w _`, `lo w e r _` |

---

## 6. Solved Illustrations

### Illustration 1: Tokenizing an Unseen Word at Inference Time
**Problem:**
Using the 4-merge table learned in Section 5 ($\mathcal{M} = [(\text{'e', 's'}), (\text{'es', 't'}), (\text{'est', '\_'}), (\text{'l', 'o'})]$), tokenize the novel word:
$$w_{\text{test}} = \text{"lowest"}$$

**Solution:**
1. Split into characters: `['l', 'o', 'w', 'e', 's', 't', '_']`.
2. Apply Merge 1 `('e', 's') \to 'es'`:
   `['l', 'o', 'w', 'es', 't', '_']`.
3. Apply Merge 2 `('es', 't') \to 'est'`:
   `['l', 'o', 'w', 'est', '_']`.
4. Apply Merge 3 `('est', '_') \to 'est_'`:
   `['l', 'o', 'w', 'est_']`.
5. Apply Merge 4 `('l', 'o') \to 'lo'`:
   `['lo', 'w', 'est_']`.
Final Tokenization: **`['lo', 'w', 'est_']`** (3 tokens instead of 7 characters!). $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

- **Why LLMs Struggle with "Strawberry":**
  When an LLM is asked *"How many r's are in strawberry?"*, it often answers 2 instead of 3. Why?
  In `tiktoken`, `"strawberry"` is a **single atomic token** (`token_id: 49609`). The transformer's self-attention never sees the individual letters `s-t-r-a-w-b-e-r-r-y`!
- **Digit Tokenization (LLaMA 3 vs. GPT-3.5):**
  Early models tokenized numbers arbitrarily (e.g., `12345` into `[12, 345]`). Modern tokenizers (LLaMA 3, GPT-4) enforce single-digit tokenization (`[1, 2, 3, 4, 5]`), vastly boosting mathematical and arithmetic performance!

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Reproduces the exact 4-step BPE training merges on the toy corpus.
   - Verifies encoding of unseen word `"lowest"` $\to$ `['lo', 'w', 'est_']`.
2. **Complete Byte-Level BPE Tokenizer from Scratch:**
   - Raw UTF-8 byte stream processing.
   - Full training loop computing pair statistics and merge dictionaries.
   - Encode and Decode functions with round-trip reconstruction guarantees ($D(E(x)) \equiv x$).

See implementation in:
[`12_modern_llm_architectures/code/01_tokenization_from_scratch.py`](./code/01_tokenization_from_scratch.py)
