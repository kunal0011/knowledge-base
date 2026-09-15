---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 451: Sort Characters By Frequency"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
  - bucket-sort
  - hash-map
  - amazon
  - google
---

# LeetCode 451: Sort Characters By Frequency

**Target Companies:** Amazon, Google, Bloomberg, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Priority Queue (Max-Heap) / Bucket Sort (Linear Time) / Hash Map

---

### Problem Statement

Given a string `s`, sort it in **decreasing order** based on the **frequency** of the characters. The **frequency** of a character is the number of times it appears in the string.

Return *the sorted string*. If there are multiple answers, return *any of them*.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str`, where $1 \le \text{len}(s) \le 5 \times 10^5$.
  - `s` consists of uppercase and lowercase English letters and digits.
- **Output:**
  - `str`: The reconstructed string where characters with higher frequencies appear earlier. Characters with identical frequencies may appear in any order.
- **Constraints:**
  - Case-sensitive (`'A'` and `'a'` are considered distinct characters).
  - All occurrences of the same character must be contiguous in the result.

---

### Key Idea & Intuition

The problem requires ordering characters purely by their occurrence count.
This breaks naturally into two phases:
1. **Frequency Counting:** Scan $s$ once to map each character to its occurrence frequency in $\mathcal{O}(N)$ time.
2. **Sorting by Frequency:**
   - **Approach 1: Max-Heap ($\mathcal{O}(N + U \log U)$ Time):**
     - Push all `(-frequency, character)` pairs into a max-heap.
     - Repeatedly pop the highest frequency character and append `char * frequency` to the result buffer.
     - Since the number of unique ASCII characters $U \le 62$ (or $\le 128$), $U \log U$ is an ultra-small constant ($\approx 62 \log_2 62 \approx 370$ operations)!
   - **Approach 2: Bucket Sort ($\mathcal{O}(N)$ Strictly Linear Time):**
     - Create an array of lists `buckets` of size $N + 1$, where `buckets[f]` stores all characters appearing with frequency $f$.
     - Iterate from frequency $N$ down to $1$, appending `char * f` for all characters in each bucket.

---

### Solution Approach (Step-by-Step)

#### Algorithm 1: Max-Heap
1. Count frequencies using `count = Counter(s)`.
2. Build max-heap: `heap = [(-freq, char) for char, freq in count.items()]`.
3. `heapify(heap)`.
4. Initialize `result = []`.
5. While `heap` is non-empty:
   - `neg_f, char = heappop(heap)`.
   - `result.append(char * (-neg_f))`.
6. Return `''.join(result)`.

#### Algorithm 2: Bucket Sort
1. Count frequencies using `count = Counter(s)`.
2. Initialize `buckets = [[] for _ in range(len(s) + 1)]`.
3. For `char, freq` in `count.items()`:
   - `buckets[freq].append(char)`.
4. Iterate $f$ from $len(s)$ down to $1$:
   - For `char` in `buckets[f]`:
     - `result.append(char * f)`.
5. Return `''.join(result)`.

---

### Visual Algorithm Walkthrough

Let $s = \text{"tree"}$:

```
Step 1: Count Frequencies
  'e': 2
  't': 1
  'r': 1

Max-Heap Approach:
  Heap after heapify:
  [ (-2, 'e'), (-1, 't'), (-1, 'r') ]

  Pop 1: (-2, 'e') -> append "ee"
  Pop 2: (-1, 't') -> append "t"
  Pop 3: (-1, 'r') -> append "r"
  Result: "eetr" (or "eert")

Bucket Sort Approach:
  Bucket 0: []
  Bucket 1: ['t', 'r']
  Bucket 2: ['e']
  Bucket 3: []
  Bucket 4: []

  Iterate from Bucket 4 down to 1:
    Bucket 2 has ['e'] -> "ee"
    Bucket 1 has ['t', 'r'] -> "t", "r"
  Result: "eetr"
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Unequal Frequencies

- **Input:** `s = "tree"`
- **Output:** `"eert"` (or `"eetr"`)

#### Example 2: Equal Frequencies

- **Input:** `s = "cccaaa"`
- **Output:** `"aaaccc"` (or `"cccaaa"`)

#### Example 3: Case Sensitivity

- **Input:** `s = "Aabb"`
- **Output:** `"bbAa"` (or `"bbaA"`, since `'b'` has count 2, while `'A'` and `'a'` each have count 1)

---

### Multi-Language Implementations

#### Python 3

##### Optimal Bucket Sort ($\mathcal{O}(N)$ Time, $\mathcal{O}(N)$ Space)
```python
from collections import Counter
from typing import List

class Solution:
    def frequencySort(self, s: str) -> str:
        count = Counter(s)
        n = len(s)
        # buckets[i] contains characters that appear exactly i times
        buckets: List[List[str]] = [[] for _ in range(n + 1)]

        for char, freq in count.items():
            buckets[freq].append(char)

        result = []
        for freq in range(n, 0, -1):
            for char in buckets[freq]:
                result.append(char * freq)

        return "".join(result)
```

##### Max-Heap Approach ($\mathcal{O}(N + U \log U)$ Time)
```python
import heapq
from collections import Counter

class SolutionHeap:
    def frequencySort(self, s: str) -> str:
        count = Counter(s)
        max_heap = [(-freq, char) for char, freq in count.items()]
        heapq.heapify(max_heap)

        result = []
        while max_heap:
            neg_freq, char = heapq.heappop(max_heap)
            result.append(char * (-neg_freq))

        return "".join(result)
```

#### C++17

```cpp
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>

class Solution {
public:
    // Optimal O(N) Bucket Sort
    std::string frequencySort(const std::string& s) {
        std::unordered_map<char, int> count;
        for (char c : s) {
            count[c]++;
        }

        int n = static_cast<int>(s.size());
        std::vector<std::string> buckets(n + 1, "");

        for (const auto& [ch, freq] : count) {
            buckets[freq].append(freq, ch);
        }

        std::string result = "";
        result.reserve(n);

        for (int f = n; f >= 1; --f) {
            if (!buckets[f].empty()) {
                result += buckets[f];
            }
        }

        return result;
    }
};
```

#### Java

```java
import java.util.*;

public class Solution {
    // Optimal O(N) Bucket Sort
    public String frequencySort(String s) {
        Map<Character, Integer> count = new HashMap<>();
        for (char c : s.toCharArray()) {
            count.put(c, count.getOrDefault(c, 0) + 1);
        }

        int n = s.length();
        List<Character>[] buckets = new List[n + 1];
        for (int i = 0; i <= n; i++) {
            buckets[i] = new ArrayList<>();
        }

        for (Map.Entry<Character, Integer> entry : count.entrySet()) {
            buckets[entry.getValue()].add(entry.getKey());
        }

        StringBuilder sb = new StringBuilder();
        for (int f = n; f >= 1; f--) {
            for (char c : buckets[f]) {
                for (int i = 0; i < f; i++) {
                    sb.append(c);
                }
            }
        }

        return sb.toString();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - **Bucket Sort:** $\mathcal{O}(N)$ where $N$ is the length of string $s$.
    - Frequency counting: $\mathcal{O}(N)$.
    - Bucket distribution: $\mathcal{O}(U)$ where $U$ is unique characters ($U \le 62$).
    - Result assembly: $\mathcal{O}(N)$.
  - **Max-Heap:** $\mathcal{O}(N + U \log U)$ which is essentially $\mathcal{O}(N)$ since $U \le 62$.
- **Space Complexity:** $\mathcal{O}(N)$ to store frequency buckets and the output string buffer.

---

### Takeaway Pattern & Interview Traps

1. **Character Contiguity:**
   - The question requires that all identical characters appear together in a single contiguous block (e.g. `"ee"` in `"eert"`). You cannot interleave characters with the same frequency (e.g., `"eter"` is invalid).
2. **Case Sensitivity:**
   - Uppercase and lowercase letters have distinct ASCII codes (`'A'` is 65, `'a'` is 97). Do not lowercase the string before counting.