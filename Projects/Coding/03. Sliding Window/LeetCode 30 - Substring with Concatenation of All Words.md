---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 30: Substring with Concatenation of All Words"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 30: Substring with Concatenation of All Words

**LeetCode 30 – Substring with Concatenation of All Words**.

This is one of the trickier sliding-window/hashmap problems.

---

## 🔹 Problem Restatement

* You’re given a string `s` and an array of words `words`.
* All words have **the same length**.
* Return all **starting indices** in `s` where the substring is a **concatenation of all words** (in any order, without extra characters).

---

## 🔹 Example

```
s = "barfoothefoobarman"
words = ["foo","bar"]
```

Valid substrings:

* `"barfoo"` starting at index `0`
* `"foobar"` starting at index `9`

Answer = `[0, 9]`.

---

## 🔹 Key Observations

1. Each word has the same length `L`.
2. The target substring length is `total_len = L * len(words)`.
3. We can slide a window of size `total_len` across `s` and check if it contains all words.
4. To check validity efficiently, use **hashmaps** for word counts.

---

## 🔹 Approaches

### **1. Brute Force**

* For each index `i` in `s`, extract substring of length `total_len`.
* Split into chunks of size `L`.
* Check if counts match `words_count`.

⏱ Complexity: **O(n \* m \* L)**, where

* `n = len(s)`
* `m = len(words)`
* `L = word length`

❌ Works but too slow for large inputs.

---

### **2. Sliding Window with HashMap (Optimal)**

* Since all words are the same length `L`, we can align windows at offsets `0…L-1`.
* For each offset, slide in steps of `L` (word size).
* Maintain a window count map of words.
* If window size > `m` words, shrink from left.
* If counts match, record the starting index.

⏱ Complexity: **O(n \* L)** (linear in `s`).  
✅ Efficient and accepted.

---

## 🔹 Implementation

### HashMap + Sliding Window

```python
from collections import Counter

def findSubstring(s, words):
    if not s or not words:
        return []

    word_len = len(words[0])
    word_count = len(words)
    total_len = word_len * word_count
    n = len(s)

    word_map = Counter(words)
    res = []

    # try each offset
    for i in range(word_len):
        left = i
        seen = Counter()
        count = 0

        for j in range(i, n - word_len + 1, word_len):
            word = s[j:j+word_len]

            if word in word_map:
                seen[word] += 1
                count += 1

                # shrink if more than expected
                while seen[word] > word_map[word]:
                    seen[s[left:left+word_len]] -= 1
                    left += word_len
                    count -= 1

                # valid window
                if count == word_count:
                    res.append(left)

            else:  # reset
                seen.clear()
                count = 0
                left = j + word_len

    return res
```

---

## 🔹 Example Walkthrough

`s = "barfoothefoobarman", words = ["foo","bar"]`

* `word_len = 3`, `word_count = 2`, `total_len = 6`.
* Offsets checked: 0, 1, 2.

Offset 0:

* `"barfoo"` → valid → index `0`.
* `"foobar"` → valid → index `9`.

Answer = `[0, 9]`.

---

## 🔹 Which Approaches Work?

* **Brute Force** → Works for small cases, but slow.
* **Sliding Window with HashMap** → Optimal.
* **Sorting** not possible since order of words inside substring doesn’t matter (only counts).