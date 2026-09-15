---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 2024: Maximize the Confusion of an Exam"
tags:
  - leetcode
  - coding
  - sliding-window
  - string
  - two-pointers
  - amazon
  - google
---

# LeetCode 2024: Maximize the Confusion of an Exam

**Target Companies:** Amazon, Google, Microsoft  
**Difficulty:** Medium  
**Topic:** Dynamic Sliding Window / Binary Flips Invariant  

---

### Problem Statement

A teacher is writing a test with `n` true/false questions, with `'T'` denoting true and `'F'` denoting false. He wants to confuse the students by maximizing the number of consecutive questions with the **same** answer (multiple consecutive `'T'`s or multiple consecutive `'F'`s).

You are given a string `answerKey`, where `answerKey[i]` is the original answer to the $i$-th question. In addition, you are given an integer `k`, the maximum number of times you may perform the following operation:
- Change the answer key for any question to `'T'` or `'F'` (i.e., set `answerKey[i]` to `'T'` or `'F'`).

Return *the **maximum** number of consecutive `'T'`s or `'F'`s in the answer key after performing the operation at most `k` times*.

---

### Input & Output Formats & Constraints

- **Input:** `answerKey: str`, `k: int`
- **Output:** `int` (maximum length of uniform consecutive answers)
- **Constraints:**
  - $n == \text{answerKey.length}$
  - $1 \le n \le 5 \times 10^4$
  - `answerKey[i]` is either `'T'` or `'F'`.
  - $1 \le k \le n$

---

### Key Idea & Intuition

- **The Window Validity Invariant:**
  - A substring $[L, R]$ can be converted into entirely `'T'`s or entirely `'F'`s using at most $k$ flips if and only if the minority character occurs at most $k$ times:
    $$\min(\text{count}('T'), \text{count}('F')) \le k$$
- **Connection to Max Consecutive Ones III (LeetCode 1004):**
  - Making all answers `'T'` is equivalent to flipping at most $k$ `'F'`s $\implies$ LC 1004 with `'F'` treated as zero.
  - Making all answers `'F'` is equivalent to flipping at most $k$ `'T'`s $\implies$ LC 1004 with `'T'` treated as zero.
- **Unified Single-Pass Sliding Window:**
  - Maintain counters `count_T` and `count_F` for the current window $[L, R]$.
  - As $R$ expands, increment the count of `answerKey[R]`.
  - While $\min(\text{count\_T}, \text{count\_F}) > k$, contract $L$ and decrement the respective counter.
  - Update $\text{max\_len} = \max(\text{max\_len}, R - L + 1)$.

---

### Solution Approach (Step-by-Step)

1. Initialize `left = 0`, `count_T = 0`, `count_F = 0`, and `max_len = 0`.
2. Iterate `right` from $0$ to $\text{len}(answerKey) - 1$:
   - If `answerKey[right] == 'T'`:
     - `count_T += 1`
   - Else:
     - `count_F += 1`
   - While $\min(\text{count\_T}, \text{count\_F}) > k$:
     - If `answerKey[left] == 'T'`:
       - `count_T -= 1`
     - Else:
       - `count_F -= 1`
     - `left += 1`
   - `max_len = max(max_len, right - left + 1)`
3. Return `max_len`.

---

### Visual Algorithm Walkthrough

Let `answerKey = "TTFTTFTT"`, $k = 1$.

```
Indices:    0  1  2  3  4  5  6  7
String:     T  T  F  T  T  F  T  T

R=0 (T): T=1, F=0, min=0 <= 1 -> len = 1
R=1 (T): T=2, F=0, min=0 <= 1 -> len = 2
R=2 (F): T=2, F=1, min=1 <= 1 -> len = 3
R=3 (T): T=3, F=1, min=1 <= 1 -> len = 4
R=4 (T): T=4, F=1, min=1 <= 1 -> len = 5
R=5 (F): T=4, F=2, min=2 > 1 (INVALID!)
         Shrink L until min(T, F) <= 1:
         - L=0 ('T'): T=3, F=2, min=2 > 1
         - L=1 ('T'): T=2, F=2, min=2 > 1
         - L=2 ('F'): T=2, F=1, min=1 <= 1!
         New L = 3. Window [3..5] ("TTF"), length = 3.
R=6 (T): T=3, F=1, min=1 <= 1 -> len = 4
R=7 (T): T=4, F=1, min=1 <= 1 -> len = 7 - 3 + 1 = 5.

Max length achieved: 5 (Window [0..4] "TTFTT" by flipping 1 'F').
```

---

### Solved Examples with Multiple Inputs

| Test Case | `answerKey` | `k` | Flip Target | Max Consecutive |
| :--- | :--- | :--- | :--- | :--- |
| **Example 1** | `"TTFF"` | `2` | Flip both `'F'`s to `'T'` or vice versa | `4` |
| **Example 2** | `"TFFT"` | `1` | Flip middle `'F'` or `'T'` | `3` |
| **Example 3** | `"TTFTTFTT"` | `1` | Flip 1st `'F'` $\implies$ `"TTTTTFTT"` | `5` |
| **All Same** | `"TTTT"` | `1` | Already all `'T'` | `4` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
        """
        Finds the maximum consecutive 'T's or 'F's after at most k changes.
        Uses a dynamic sliding window tracking the count of both characters.
        """
        left = 0
        counts = {'T': 0, 'F': 0}
        max_len = 0

        for right, ch in enumerate(answerKey):
            counts[ch] += 1

            # The minority character represents the flips needed.
            # While flips required exceed k, contract from left.
            while min(counts['T'], counts['F']) > k:
                counts[answerKey[left]] -= 1
                left += 1

            max_len = max(max_len, right - left + 1)

        return max_len
```

#### C++17
```cpp
#include <string>
#include <algorithm>

class Solution {
public:
    int maxConsecutiveAnswers(const std::string& answerKey, int k) {
        int left = 0;
        int count_t = 0;
        int count_f = 0;
        int max_len = 0;

        for (int right = 0; right < static_cast<int>(answerKey.size()); ++right) {
            if (answerKey[right] == 'T') {
                count_t++;
            } else {
                count_f++;
            }

            while (std::min(count_t, count_f) > k) {
                if (answerKey[left] == 'T') {
                    count_t--;
                } else {
                    count_f--;
                }
                left++;
            }

            max_len = std::max(max_len, right - left + 1);
        }

        return max_len;
    }
};
```

#### Java
```java
class Solution {
    public int maxConsecutiveAnswers(String answerKey, int k) {
        int left = 0;
        int countT = 0;
        int countF = 0;
        int maxLen = 0;
        int n = answerKey.length();

        for (int right = 0; right < n; right++) {
            if (answerKey.charAt(right) == 'T') {
                countT++;
            } else {
                countF++;
            }

            while (Math.min(countT, countF) > k) {
                if (answerKey.charAt(left) == 'T') {
                    countT--;
                } else {
                    countF--;
                }
                left++;
            }

            maxLen = Math.max(maxLen, right - left + 1);
        }

        return maxLen;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ where $N = \text{answerKey.length}$.
  - The `right` pointer iterates through the string once.
  - The `left` pointer advances at most $N$ times total.
  - Constant number of operations per element, running in $< 10 \text{ ms}$ for $N = 5 \times 10^4$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space since we only maintain integer counters for `'T'` and `'F'`.

---

### Takeaway Pattern & Interview Traps

- **The Minority Element Invariant:** Rather than writing two separate loops (one for maximizing `'T'`s and one for maximizing `'F'`s), the condition $\min(\text{count}(T), \text{count}(F)) \le k$ solves both directions simultaneously in one unified pass.
- **Contrast with LC 424:** In LC 424 (any uppercase English letter), the condition is `window_size - max_freq <= k`. Here, since there are only 2 possible characters, $\text{window\_size} - \max(\text{freq}) = \min(\text{freq})$.