---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 767: Reorganize String"
tags:
  - leetcode
  - coding
  - greedy
  - heap
  - hash-table
  - string
  - amazon
  - meta
  - google
  - microsoft
---

# LeetCode 767: Reorganize String

**Target Companies:** Amazon, Meta, Google, Microsoft, Uber, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Hash Table / String / Heap (Priority Queue)  

---

### Problem Statement

Given a string `s`, rearrange the characters of `s` so that any two adjacent characters are not the same.

Return *any possible rearrangement of `s` or return `""` if not possible*.

---

### Input & Output Formats & Constraints

- **Input:** A string `s` ($1 \le |s| \le 500$) consisting of lowercase English letters.
- **Output:** A valid reorganized string, or `""` if impossible.
- **Constraints:**
  - `1 <= s.length <= 500`
  - `s` consists of lowercase English letters.

---

### Key Idea & Intuition

#### The Pigeonhole Feasibility Condition
Let $N = |s|$, and let $f_{\max}$ be the maximum frequency of any character in `s`.
In any valid rearrangement where no two adjacent characters are identical:
- The most frequent character must be separated by at least one other character.
- Placing this character at alternate indices ($0, 2, 4, \dots$) accommodates at most $\lfloor (N + 1) / 2 \rfloor$ copies.
- If $f_{\max} > \lfloor (N + 1) / 2 \rfloor$, by the Pigeonhole Principle, at least two copies of this character must be adjacent. Thus, no valid reorganization exists, and we immediately return `""`.

#### Greedy Choice via Max-Heap
When $f_{\max} \le \lfloor (N + 1) / 2 \rfloor$, a solution is always guaranteed.
To prevent being cornered into adjacent duplicates near the end of the construction:
1. Always prioritize placing the character with the **highest remaining frequency**.
2. To avoid repeating the same character back-to-back, we greedily pair the **top two most frequent characters** from a max-heap in each iteration:
   - Pop character $c_1$ with count $cnt_1$.
   - Pop character $c_2$ with count $cnt_2$.
   - Append $c_1$ followed by $c_2$ to the output.
   - Decrement their counts and push back any character whose count is still $> 0$.
3. If exactly one character remains in the heap when the loop terminates, its count must be 1 (guaranteed by the feasibility check). Append it to complete the string.

---

### Solution Approach (Step-by-Step)

1. **Count Frequencies:**
   - Count character frequencies using a frequency array or hash map.
2. **Feasibility Check:**
   - Find $\max(f)$ across all characters.
   - If $\max(f) > (N + 1) / 2$, return `""`.
3. **Build Max-Heap:**
   - Push pairs `(frequency, char)` into a max-heap (ordered by descending frequency).
4. **Greedy Pairing Loop:**
   - While heap size $\ge 2$:
     - Pop $(cnt_1, ch_1)$ and $(cnt_2, ch_2)$.
     - Append $ch_1$ then $ch_2$ to the result buffer.
     - If $cnt_1 - 1 > 0$, push $(cnt_1 - 1, ch_1)$ back into the heap.
     - If $cnt_2 - 1 > 0$, push $(cnt_2 - 1, ch_2)$ back into the heap.
5. **Handle Odd Remainder:**
   - If 1 element remains in the heap, pop and append it.
6. **Return Output:**
   - Convert buffer to string and return.

---

### Visual Algorithm Walkthrough

#### Trace for $s = \text{"aaabbc"}$ ($N = 6$)
```
Counts: a: 3, b: 2, c: 1
Max count = 3 <= (6 + 1) / 2 = 3 -> Feasible!

Initial Max-Heap: [(3, 'a'), (2, 'b'), (1, 'c')]

Round 1:
- Pop top 2: ('a', 3) and ('b', 2)
- Append: "ab"
- Remaining counts: 'a': 2, 'b': 1
- Push back -> Heap: [(2, 'a'), (1, 'b'), (1, 'c')]

Round 2:
- Pop top 2: ('a', 2) and ('b', 1)
- Append: "ab" -> Buffer: "abab"
- Remaining counts: 'a': 1, 'b': 0
- Push back -> Heap: [(1, 'a'), (1, 'c')]

Round 3:
- Pop top 2: ('a', 1) and ('c', 1)
- Append: "ac" -> Buffer: "ababac"
- Remaining counts: 'a': 0, 'c': 0
- Heap is empty.

Final Result: "ababac" (Valid: no two adjacent identical characters)
```

#### Trace for $s = \text{"aaab"}$ ($N = 4$)
```
Counts: a: 3, b: 1
Max count = 3 > (4 + 1) / 2 = 2 -> Infeasible!
Immediate Return: ""
```

---

### Solved Examples with Multiple Inputs

| Input $s$ | Counts | Feasibility Check ($\le \lfloor (N+1)/2 \rfloor$) | Heap Pairing Sequence | Output |
|---|---|---|---|---|
| `"aab"` | a:2, b:1 | $2 \le 2$ (Valid) | Round 1: 'a', 'b'; Remainder: 'a' | `"aba"` |
| `"aaab"` | a:3, b:1 | $3 > 2$ (Invalid) | N/A | `""` |
| `"vvvlo"` | v:3, l:1, o:1 | $3 \le 3$ (Valid) | ('v','l'), ('v','o'), ('v') | `"vlvov"` |
| `"baaba"` | a:3, b:2 | $3 \le 3$ (Valid) | ('a','b'), ('a','b'), ('a') | `"ababa"` |

---

### Multi-Language Implementations

#### Python 3
```python
import heapq
from collections import Counter

class Solution:
    def reorganizeString(self, s: str) -> str:
        freq = Counter(s)
        n = len(s)
        max_freq = max(freq.values())
        
        # Invariant: Impossible if most frequent char exceeds ceil(n / 2)
        if max_freq > (n + 1) // 2:
            return ""
            
        # Max-heap: store (-count, char)
        max_heap = [(-count, ch) for ch, count in freq.items()]
        heapq.heapify(max_heap)
        
        result: list[str] = []
        
        # Greedily pair top two distinct frequent characters
        while len(max_heap) >= 2:
            cnt1, ch1 = heapq.heappop(max_heap)
            cnt2, ch2 = heapq.heappop(max_heap)
            
            result.append(ch1)
            result.append(ch2)
            
            if cnt1 + 1 < 0:
                heapq.heappush(max_heap, (cnt1 + 1, ch1))
            if cnt2 + 1 < 0:
                heapq.heappush(max_heap, (cnt2 + 1, ch2))
                
        # Append final lone character if odd length
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

class Solution {
public:
    std::string reorganizeString(const std::string& s) {
        int n = static_cast<int>(s.size());
        int count[26] = {0};
        int max_freq = 0;
        
        for (char ch : s) {
            count[ch - 'a']++;
            max_freq = std::max(max_freq, count[ch - 'a']);
        }
        
        if (max_freq > (n + 1) / 2) {
            return "";
        }
        
        // Max-heap storing pair<frequency, char>
        std::priority_queue<std::pair<int, char>> pq;
        for (int i = 0; i < 26; ++i) {
            if (count[i] > 0) {
                pq.push({count[i], static_cast<char>('a' + i)});
            }
        }
        
        std::string result = "";
        result.reserve(n);
        
        while (pq.size() >= 2) {
            auto [cnt1, ch1] = pq.top(); pq.pop();
            auto [cnt2, ch2] = pq.top(); pq.pop();
            
            result.push_back(ch1);
            result.push_back(ch2);
            
            if (--cnt1 > 0) pq.push({cnt1, ch1});
            if (--cnt2 > 0) pq.push({cnt2, ch2});
        }
        
        if (!pq.empty()) {
            result.push_back(pq.top().second);
        }
        
        return result;
    }
};
```

#### Java 17
```java
import java.util.PriorityQueue;

class Solution {
    public String reorganizeString(String s) {
        int n = s.length();
        int[] count = new int[26];
        int maxFreq = 0;
        
        for (int i = 0; i < n; i++) {
            int idx = s.charAt(i) - 'a';
            count[idx]++;
            maxFreq = Math.max(maxFreq, count[idx]);
        }
        
        // Feasibility check
        if (maxFreq > (n + 1) / 2) {
            return "";
        }
        
        // Max-heap ordered by frequency descending
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> Integer.compare(b[1], a[1]));
        for (int i = 0; i < 26; i++) {
            if (count[i] > 0) {
                pq.offer(new int[]{i, count[i]});
            }
        }
        
        StringBuilder sb = new StringBuilder(n);
        
        while (pq.size() >= 2) {
            int[] first = pq.poll();
            int[] second = pq.poll();
            
            sb.append((char) ('a' + first[0]));
            sb.append((char) ('a' + second[0]));
            
            if (--first[1] > 0) pq.offer(first);
            if (--second[1] > 0) pq.offer(second);
        }
        
        if (!pq.isEmpty()) {
            sb.append((char) ('a' + pq.poll()[0]));
        }
        
        return sb.toString();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \log |\Sigma|)$, where $N = |s|$ and $\Sigma$ is the alphabet size ($|\Sigma| \le 26$). Since the heap contains at most 26 entries, heap operations take $\mathcal{O}(\log 26) = \mathcal{O}(1)$ time. Thus, the total time complexity is strictly $\mathcal{O}(N)$.
- **Space Complexity:** $\mathcal{O}(|\Sigma|) = \mathcal{O}(1)$ auxiliary space for the frequency map and priority queue (at most 26 elements).

---

### Takeaway Pattern & Interview Traps

1. **Alternate Approach (Direct Even/Odd Index Placement in $\mathcal{O}(N)$):**
   - We can place the most frequent character on even indices `0, 2, 4, ...` first.
   - Once the most frequent character is placed, fill remaining characters continuously across even indices, wrapping around to odd indices `1, 3, 5, ...` when even positions run out.
   - Because $f_{\max} \le (N + 1) / 2$, the most frequent character will never wrap into odd positions adjacent to itself!
2. **Ceil Division Trick:** Checking $\max(f) > (N + 1) / 2$ handles both even and odd string lengths without floating point arithmetic (e.g. for $N = 5$, $(5+1)/2 = 3$; for $N = 4$, $(4+1)/2 = 2$).
3. **Pop Two at a Time:** Trying to pop only one character and caching the "previous" character works as well, but popping two at once eliminates edge cases regarding whether the popped character matches the immediately preceding character.