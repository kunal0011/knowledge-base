---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 466: Count The Repetitions"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - cycle-detection
  - pigeonhole-principle
  - math
  - google
---

# LeetCode 466: Count The Repetitions

**Target Companies:** Google, Meta, Microsoft  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / Cycle Detection / Pigeonhole Principle / String Matching  

---

### Problem Statement

We define `str = [s, n]` as the string `s` concatenated and repeated `n` times.

- For example, `str = ["abc", 3] == "abcabcabc"`.

We say that string `s1` can be obtained from string `s2` if we can remove some characters from `s2` such that it becomes `s1`.

You are given two strings `s1` and `s2` and two integers `n1` and `n2`. You have the two strings `str1 = [s1, n1]` and `str2 = [s2, n2]`.

Return the maximum integer `m` such that `str = [str2, m]` can be obtained from `str1` as a subsequence.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s1: str`, `n1: int`
  - `s2: str`, `n2: int`
- **Output:**
  - `int` — Maximum multiplier $M$ such that $[S_2, M]$ is a subsequence of $[S_1, 1]$.
- **Constraints:**
  - $1 \le s_1.\text{length}, s_2.\text{length} \le 100$
  - $1 \le n_1, n_2 \le 10^6$
  - $s_1$ and $s_2$ consist of lowercase English letters.

---

### Key Idea & Intuition

1. **Why Direct Simulation Fails:**
   - The expanded string $[s_1, n_1]$ has length up to $100 \times 10^6 = 10^8$. Materializing or greedily scanning all $10^8$ characters will exceed memory and time limits.
   - However, $s_1$ and $s_2$ are very short ($\le 100$).

2. **State Compression via Subsequence Pointers:**
   - As we scan copies of $s_1$ one by one, we track:
     - `index`: the current index in $s_2$ that we are trying to match ($0 \le \text{index} < |s_2|$).
     - `count`: total number of times $s_2$ has been completely matched so far.
   - At the end of each $s_1$ block, we record the state `index` in $s_2$.

3. **Pigeonhole Principle & Cycle Detection:**
   - The pointer `index` in $s_2$ can take only $|s_2|$ distinct values ($0$ to $|s_2| - 1$).
   - By the **Pigeonhole Principle**, within at most $|s_2| + 1$ repetitions of $s_1$, the ending `index` **must repeat**!
   - Once an `index` repeats:
     - A periodic cycle is established!
     - Let the first occurrence of this `index` happen after $k_1$ copies of $s_1$ with $c_1$ matched $s_2$ strings.
     - Let the second occurrence happen after $k_2$ copies of $s_1$ with $c_2$ matched $s_2$ strings.
     - **Cycle length in $s_1$ blocks:** $\Delta k = k_2 - k_1$.
     - **Matches gained per cycle:** $\Delta c = c_2 - c_1$.
   - We can fast-forward through the remaining $s_1$ blocks:
     $$\text{num\_cycles} = \frac{n_1 - k_1}{\Delta k}$$
     $$\text{total\_matches} += \text{num\_cycles} \times \Delta c$$
     $$\text{remaining\_blocks} = (n_1 - k_1) \pmod{\Delta k}$$
   - Finally, simulate the remaining $s_1$ blocks and divide the total matched $s_2$ count by $n_2$ to obtain $M$.

---

### Solution Approach (Step-by-Step)

1. **Track Block History:**
   - Maintain `repeat_count[k]`: total completed $s_2$ matches after scanning $k$ copies of $s_1$.
   - Maintain `next_index[k]`: the index in $s_2$ reached after scanning $k$ copies of $s_1$.
2. **Iterate Through $s_1$ Blocks ($k = 1 \dots n_1$):**
   - For each character in $s_1$:
     - If character matches $s_2[\text{index}]$, advance $\text{index} += 1$.
     - If $\text{index} == |s_2|$, increment match count and reset $\text{index} = 0$.
   - Save `repeat_count[k]` and `next_index[k]`.
   - **Check for Cycle:** Check if $\text{next\_index}[k]$ was seen previously at some $start < k$:
     - If seen:
       - Cycle period: `period = k - start`.
       - Matches per cycle: `cycle_count = repeat_count[k] - repeat_count[start]`.
       - Fast-forward cycles: `remaining_k = n1 - k`.
       - `cycles = remaining_k // period`.
       - `total_s2 = repeat_count[k] + cycles * cycle_count`.
       - Add remaining tail matches: `repeat_count[start + (remaining_k % period)] - repeat_count[start]`.
       - Return `total_s2 // n2`.
3. **If No Cycle Detected (when $n_1 \le |s_2|$):**
   - Return `repeat_count[n1] // n2`.

---

### Visual Algorithm Walkthrough

Suppose $s_1 = \text{"acb"}$, $n_1 = 4$, $s_2 = \text{"ab"}$, $n_2 = 2$.

```
Block 1 (s1 = "acb"):
  'a' matches s2[0] -> index = 1
  'c' mismatch -> skip
  'b' matches s2[1] -> s2 completed! count = 1, index = 0
  next_index[1] = 0, repeat_count[1] = 1

Block 2 (s1 = "acb"):
  'a' matches s2[0] -> index = 1
  'c' mismatch -> skip
  'b' matches s2[1] -> s2 completed! count = 2, index = 0
  next_index[2] = 0, repeat_count[2] = 2

Cycle detected at k = 2!
  Previous occurrence was at start = 1 (both have next_index = 0).
  period = 2 - 1 = 1 block of s1
  cycle_count = repeat_count[2] - repeat_count[1] = 2 - 1 = 1 copy of s2
  Remaining blocks = n1 - 2 = 4 - 2 = 2
  Fast-forward: 2 cycles * 1 = +2 copies of s2
  Total s2 copies = repeat_count[2] + 2 = 2 + 2 = 4

Final Answer:
  total_s2 // n2 = 4 // 2 = 2
```

---

### Solved Examples with Multiple Inputs

| Case | `s1, n1` | `s2, n2` | Total $s_2$ Matched | Result ($M$) | Explanation |
|---|---|---|---|---|---|
| **Standard Cycle** | `"acb", 4` | `"ab", 2` | `4` | `2` | $[s_2, 2] = \text{"abab"}$ fits in $\text{"acbacbacbacb"}$ |
| **No Matches** | `"abc", 10` | `"xyz", 1` | `0` | `0` | Characters of $s_2$ don't exist in $s_1$ |
| **Direct Single Block** | `"aaa", 3` | `"aa", 1` | `4` | `4` | $9$ 'a's contain 4 pairs of `"aa"` |
| **Large Count** | `"baba", 10000` | `"ba", 2` | `20000` | `10000` | Perfect doubling per block |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import Dict

class Solution:
    def getMaxRepetitions(self, s1: str, n1: int, s2: str, n2: int) -> int:
        len1, len2 = len(s1), len(s2)
        
        # repeat_count[k] = total completed s2 after k copies of s1
        repeat_count = [0] * (n1 + 1)
        # next_index[k] = index in s2 to match after k copies of s1
        next_index = [0] * (n1 + 1)
        
        # seen maps s2_index -> s1_count k where this index was previously seen
        seen: Dict[int, int] = {}
        
        j = 0
        count = 0
        
        for k in range(1, n1 + 1):
            for ch in s1:
                if ch == s2[j]:
                    j += 1
                    if j == len2:
                        count += 1
                        j = 0
                        
            repeat_count[k] = count
            next_index[k] = j
            
            # Check for cycle
            if j in seen:
                prev_k = seen[j]
                period = k - prev_k
                cycle_count = repeat_count[k] - repeat_count[prev_k]
                
                remaining_k = n1 - k
                cycles = remaining_k // period
                rem = remaining_k % period
                
                total_s2 = repeat_count[k] + cycles * cycle_count + (repeat_count[prev_k + rem] - repeat_count[prev_k])
                return total_s2 // n2
                
            seen[j] = k
            
        return repeat_count[n1] // n2
```

#### 2. C++ (C++17 / STL)
```cpp
#include <string>
#include <vector>
#include <unordered_map>

class Solution {
public:
    int getMaxRepetitions(std::string s1, int n1, std::string s2, int n2) {
        int len1 = s1.size();
        int len2 = s2.size();

        std::vector<int> repeat_count(n1 + 1, 0);
        std::vector<int> next_index(n1 + 1, 0);
        std::unordered_map<int, int> seen;

        int j = 0;
        int count = 0;

        for (int k = 1; k <= n1; ++k) {
            for (char ch : s1) {
                if (ch == s2[j]) {
                    j++;
                    if (j == len2) {
                        count++;
                        j = 0;
                    }
                }
            }

            repeat_count[k] = count;
            next_index[k] = j;

            if (seen.find(j) != seen.end()) {
                int prev_k = seen[j];
                int period = k - prev_k;
                int cycle_count = repeat_count[k] - repeat_count[prev_k];

                int remaining_k = n1 - k;
                int cycles = remaining_k / period;
                int rem = remaining_k % period;

                int total_s2 = repeat_count[k] + cycles * cycle_count + (repeat_count[prev_k + rem] - repeat_count[prev_k]);
                return total_s2 / n2;
            }

            seen[j] = k;
        }

        return repeat_count[n1] / n2;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public int getMaxRepetitions(String s1, int n1, String s2, int n2) {
        int len1 = s1.length();
        int len2 = s2.length();

        int[] repeatCount = new int[n1 + 1];
        int[] nextIndex = new int[n1 + 1];
        Map<Integer, Integer> seen = new HashMap<>();

        int j = 0;
        int count = 0;

        for (int k = 1; k <= n1; k++) {
            for (int i = 0; i < len1; i++) {
                if (s1.charAt(i) == s2.charAt(j)) {
                    j++;
                    if (j == len2) {
                        count++;
                        j = 0;
                    }
                }
            }

            repeatCount[k] = count;
            nextIndex[k] = j;

            if (seen.containsKey(j)) {
                int prevK = seen.get(j);
                int period = k - prevK;
                int cycleCount = repeatCount[k] - repeatCount[prevK];

                int remainingK = n1 - k;
                int cycles = remainingK / period;
                int rem = remainingK % period;

                int totalS2 = repeatCount[k] + cycles * cycleCount + (repeatCount[prevK + rem] - repeatCount[prevK]);
                return totalS2 / n2;
            }

            seen.put(j, k);
        }

        return repeatCount[n1] / n2;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(|s_1| \times \min(n_1, |s_2| + 1))$  
  Because there are only $|s_2|$ distinct values for index $j$, a cycle is mathematically guaranteed to be discovered after at most $|s_2| + 1$ iterations. Each iteration scans $s_1$ in $\mathcal{O}(|s_1|)$ time. For $|s_1|, |s_2| \le 100$, maximum operations $\le 100 \times 101 \approx 10^4$ ($< 2$ ms), **completely independent of $n_1 = 10^6$**!
- **Space Complexity:** $\mathcal{O}(|s_2|)$  
  The hash map and tracking tables only need to store at most $|s_2| + 1$ entries.

---

### Takeaway Pattern & Interview Traps

1. **Pigeonhole Principle in Periodic Sequences:**
   - Whenever an astronomical repetition count ($n_1 = 10^6$) acts on a small discrete finite state space ($|s_2| \le 100$), never simulate all $n_1$ steps. Look for periodic cycles using the Pigeonhole Principle.
2. **Remainder Calculation:**
   - The remaining tail after whole cycles are skipped must be evaluated from the stored table: `repeat_count[prev_k + rem] - repeat_count[prev_k]`. Never forget this partial cycle offset.