---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 767: Reorganize String"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
  - greedy
  - string
  - amazon
  - google
---

# LeetCode 767: Reorganize String

**Target Companies:** Amazon (Top Classic), Google, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Priority Queue (Max-Heap) / Greedy Character Interleaving

---

### Problem Statement

Given a string `s`, rearrange the characters of `s` so that any two adjacent characters are not the same.

Return *any possible rearrangement of `s` or return `""` if not possible*.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str`, consisting of lowercase English letters ($1 \le \text{len}(s) \le 500$).
- **Output:**
  - `str`: Any valid rearranged string with no adjacent duplicates, or `""` if impossible.
- **Constraints:**
  - Lowercase English letters only.
  - Feasibility strictly bounded by the Pigeonhole Principle.

---

### Key Idea & Intuition

#### 1. The Mathematical Feasibility Condition (Pigeonhole Principle):
Let $n = \text{len}(s)$, and let $M$ be the frequency of the most common character.
Even if we place this most frequent character at every alternating index ($0, 2, 4, \dots$), the maximum number of non-adjacent slots available is:
$$\text{max\_allowed} = \left\lfloor \frac{n + 1}{2} \right\rfloor$$
If $M > \frac{n + 1}{2}$, it is mathematically impossible to place all copies without at least two of them touching. We must immediately return `""`.

#### 2. Greedy Max-Heap Strategy:
To guarantee that the most frequent characters do not get forced into consecutive positions near the end:
- Always pair the **two currently most frequent distinct characters** together at each step!
- Maintain a **Max-Heap** of pairs: `(-freq, char)`.
- While the heap has at least 2 elements:
  - Pop the most frequent character `(f1, c1)`.
  - Pop the second most frequent character `(f2, c2)`.
  - Append `c1` then `c2` to the result string.
  - Decrement their remaining counts by 1.
  - Push back any character that still has count $> 0$.
- If 1 character remains at the end, pop and append it (it is guaranteed to have frequency 1 by our initial feasibility check).

---

### Solution Approach (Step-by-Step)

1. Compute `count = Counter(s)`.
2. Check if $\max(count.values()) > (len(s) + 1) // 2$: return `""`.
3. Construct max-heap: `max_heap = [(-freq, char) for char, freq in count.items()]`. `heapify(max_heap)`.
4. Initialize `result = []`.
5. While `len(max_heap) >= 2`:
   - `f1, c1 = heappop(max_heap)`
   - `f2, c2 = heappop(max_heap)`
   - `result.extend([c1, c2])`
   - If `f1 + 1 < 0`: `heappush(max_heap, (f1 + 1, c1))`
   - If `f2 + 1 < 0`: `heappush(max_heap, (f2 + 1, c2))`
6. If `max_heap` is non-empty:
   - `_, c = heappop(max_heap)`
   - `result.append(c)`
7. Return `''.join(result)`.

---

### Visual Algorithm Walkthrough

Let $s = \text{"aab"}$:

```
Counts: {'a': 2, 'b': 1}, n = 3.
max_freq (2) <= (3 + 1) // 2 = 2 -> Valid!

Max-Heap:
  [ (-2, 'a'), (-1, 'b') ]

Step 1:
  Pop #1: (-2, 'a')
  Pop #2: (-1, 'b')
  Append 'a', then 'b' -> result = ['a', 'b']
  Remaining: 'a' has 1 left (-2 + 1 = -1). 'b' has 0 left.
  Re-insert 'a'.
  Heap: [ (-1, 'a') ]

Step 2:
  Heap size is 1 (< 2). Exit loop.
  Final pop: append 'a'.
  Result: ['a', 'b', 'a']

Output: "aba"
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Feasible Case

- **Input:** `s = "aab"`
- **Output:** `"aba"`

#### Example 2: Infeasible Case (Pigeonhole Violation)

- **Input:** `s = "aaab"`
- **Tracing:** $n = 4, freq(\text{'a'}) = 3$. Threshold $= (4 + 1) // 2 = 2$. $3 > 2 \implies$ Impossible.
- **Output:** `""`

#### Example 3: Multiple High-Frequency Letters

- **Input:** `s = "vvvlo"`
- **Tracing:**
  - $v: 3, l: 1, o: 1, n = 5$. Threshold $= 3 \le 3$ (Valid).
  - Pair $v$ with $l \implies$ `"vl"`.
  - Pair $v$ with $o \implies$ `"vlvo"`.
  - Final remaining $v \implies$ `"vlvov"`.
- **Output:** `"vlvov"`

---

### Multi-Language Implementations

#### Python 3

```python
import heapq
from collections import Counter

class Solution:
    def reorganizeString(self, s: str) -> str:
        count = Counter(s)
        n = len(s)

        # Mathematical feasibility check
        if max(count.values()) > (n + 1) // 2:
            return ""

        # Max-heap storing (-freq, char)
        max_heap = [(-freq, char) for char, freq in count.items()]
        heapq.heapify(max_heap)

        result = []
        while len(max_heap) >= 2:
            f1, c1 = heapq.heappop(max_heap)
            f2, c2 = heapq.heappop(max_heap)

            result.append(c1)
            result.append(c2)

            if f1 + 1 < 0:
                heapq.heappush(max_heap, (f1 + 1, c1))
            if f2 + 1 < 0:
                heapq.heappush(max_heap, (f2 + 1, c2))

        # If a single character remains, it has frequency 1
        if max_heap:
            result.append(max_heap[0][1])

        return "".join(result)
```

#### C++17

```cpp
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>

class Solution {
public:
    std::string reorganizeString(std::string s) {
        std::vector<int> freq(26, 0);
        int max_freq = 0;
        for (char c : s) {
            freq[c - 'a']++;
            max_freq = std::max(max_freq, freq[c - 'a']);
        }

        int n = static_cast<int>(s.size());
        if (max_freq > (n + 1) / 2) {
            return "";
        }

        // Max-heap storing pair of (count, char)
        std::priority_queue<std::pair<int, char>> max_heap;
        for (int i = 0; i < 26; ++i) {
            if (freq[i] > 0) {
                max_heap.emplace(freq[i], static_cast<char>('a' + i));
            }
        }

        std::string result = "";
        result.reserve(n);

        while (max_heap.size() >= 2) {
            auto [f1, c1] = max_heap.top(); max_heap.pop();
            auto [f2, c2] = max_heap.top(); max_heap.pop();

            result.push_back(c1);
            result.push_back(c2);

            if (f1 - 1 > 0) max_heap.emplace(f1 - 1, c1);
            if (f2 - 1 > 0) max_heap.emplace(f2 - 1, c2);
        }

        if (!max_heap.empty()) {
            result.push_back(max_heap.top().second);
        }

        return result;
    }
};
```

#### Java

```java
import java.util.PriorityQueue;

public class Solution {
    public String reorganizeString(String s) {
        int[] freq = new int[26];
        int maxFreq = 0;
        for (char c : s.toCharArray()) {
            freq[c - 'a']++;
            maxFreq = Math.max(maxFreq, freq[c - 'a']);
        }

        int n = s.length();
        if (maxFreq > (n + 1) / 2) {
            return "";
        }

        // Max-heap storing [count, char_code]
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>(
            (a, b) -> Integer.compare(b[0], a[0])
        );

        for (int i = 0; i < 26; i++) {
            if (freq[i] > 0) {
                maxHeap.offer(new int[]{freq[i], i});
            }
        }

        StringBuilder sb = new StringBuilder();

        while (maxHeap.size() >= 2) {
            int[] first = maxHeap.poll();
            int[] second = maxHeap.poll();

            sb.append((char) ('a' + first[1]));
            sb.append((char) ('a' + second[1]));

            if (first[0] - 1 > 0) {
                maxHeap.offer(new int[]{first[0] - 1, first[1]});
            }
            if (second[0] - 1 > 0) {
                maxHeap.offer(new int[]{second[0] - 1, second[1]});
            }
        }

        if (!maxHeap.isEmpty()) {
            sb.append((char) ('a' + maxHeap.poll()[1]));
        }

        return sb.toString();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \log \Sigma) = \mathcal{O}(N)$
  - Frequency counting takes $\mathcal{O}(N)$ where $N$ is the length of $s$.
  - Building and extracting from a heap with at most $\Sigma = 26$ distinct letters takes $\mathcal{O}(N \log 26) = \mathcal{O}(N)$ time.
- **Space Complexity:** $\mathcal{O}(\Sigma) = \mathcal{O}(1)$
  - The heap and frequency table store at most 26 elements.

---

### Takeaway Pattern & Interview Traps

1. **Pigeonhole Early Exit:**
   - Always evaluate the mathematical threshold $(n + 1) // 2$ upfront. Returning `""` immediately avoids unnecessary heap construction and guarantees that the trailing element in the heap will never have count $> 1$.
2. **Pairing Top Two:**
   - Never pop only one element and try to compare it against the previous character placed in `result`. By popping the top two distinct characters simultaneously, you mathematically guarantee alternation without special-case back-tracking.