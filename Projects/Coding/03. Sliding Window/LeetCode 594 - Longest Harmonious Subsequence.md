---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 594: Longest Harmonious Subsequence"
tags:
  - leetcode
  - coding
  - sliding-window
  - hash-map
  - counting
  - two-pointers
  - amazon
  - google
---

# LeetCode 594: Longest Harmonious Subsequence

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Easy  
**Topic:** Sliding Window / Hash Map / Counting  

---

### Problem Statement

We define a harmonious array as an array where the difference between its maximum value and its minimum value is **exactly** 1.

Given an integer array `nums`, return the length of its longest harmonious subsequence among all its possible subsequences.

A **subsequence** of an array is a sequence that can be derived from the array by deleting some or no elements without changing the order of the remaining elements.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{nums.length} \le 2 \times 10^4$).
- **Output:**
  - `int` — the length of the longest harmonious subsequence, or `0` if no such subsequence exists.
- **Constraints:**
  - $1 \le \text{nums.length} \le 2 \times 10^4$
  - $-10^9 \le \text{nums}[i] \le 10^9$

---

### Key Idea & Intuition

Because the problem asks for a **subsequence** (and not a contiguous subarray), the relative order of elements does not matter when calculating lengths.

A valid harmonious subsequence must have:
$$\max(S) - \min(S) = 1$$
This implies:
1. The subsequence can contain **at most two distinct values**: some integer $x$ and $x + 1$.
2. **Both** values $x$ and $x + 1$ must be present at least once (otherwise $\max - \min = 0 \ne 1$).
3. To maximize length, the subsequence should contain **all** occurrences of $x$ and **all** occurrences of $x + 1$ from the original array.

#### Approach 1: Hash Map Frequency Counting ($\mathcal{O}(n)$ time, $\mathcal{O}(n)$ space)
- Count frequency of every element using a hash map `freq`.
- For each unique key $x$ in `freq`:
  - If $x + 1$ is also in `freq`:
    - Length is `freq[x] + freq[x + 1]`.
    - Update `max_len = max(max_len, freq[x] + freq[x + 1])`.

#### Approach 2: Sorting + Two-Pointer Sliding Window ($\mathcal{O}(n \log n)$ time, $\mathcal{O}(1)$ auxiliary space)
- Sort `nums` in ascending order.
- Maintain a sliding window $[l, r]$:
  - While `nums[r] - nums[l] > 1`, increment `l`.
  - If `nums[r] - nums[l] == 1`, update `max_len = max(max_len, r - l + 1)`.

---

### Solution Approach (Step-by-Step: Hash Map)

1. Build frequency map `counts` of `nums`.
2. Initialize `max_len = 0`.
3. For each number `x` in `counts`:
   - If `x + 1` is in `counts`:
     - `max_len = max(max_len, counts[x] + counts[x + 1])`
4. Return `max_len`.

---

### Visual Algorithm Walkthrough

For `nums = [1, 3, 2, 2, 5, 2, 3, 7]`:

```
Frequency Map:
  1: count 1
  2: count 3
  3: count 2
  5: count 1
  7: count 1

Check pairs (x, x + 1):
  x = 1: 1 + 1 = 2 exists! count(1) + count(2) = 1 + 3 = 4. max_len = 4
  x = 2: 2 + 1 = 3 exists! count(2) + count(3) = 3 + 2 = 5. max_len = 5
  x = 3: 3 + 1 = 4 does not exist.
  x = 5: 5 + 1 = 6 does not exist.
  x = 7: 7 + 1 = 8 does not exist.

Global Maximum = 5 (subsequence: [3, 2, 2, 2, 3]).
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [1, 3, 2, 2, 5, 2, 3, 7]`
- **Output:** `5`

#### Example 2:
- **Input:** `nums = [1, 2, 3, 4]`
- **Tracing:** Pairs: (1,2) len 2; (2,3) len 2; (3,4) len 2.
- **Output:** `2`

#### Example 3 (All elements identical):
- **Input:** `nums = [1, 1, 1, 1]`
- **Tracing:** Difference between max and min is $1 - 1 = 0 \ne 1$.
- **Output:** `0`

---

### Multi-Language Implementations

#### Python 3
```python
from collections import Counter
from typing import List

class Solution:
    def findLHS(self, nums: List[int]) -> int:
        freq = Counter(nums)
        max_len = 0
        
        for x, count in freq.items():
            if x + 1 in freq:
                max_len = max(max_len, count + freq[x + 1])
                
        return max_len
```

#### C++17
```cpp
#include <vector>
#include <unordered_map>
#include <algorithm>

class Solution {
public:
    int findLHS(const std::vector<int>& nums) {
        std::unordered_map<int, int> freq;
        for (int num : nums) {
            freq[num]++;
        }
        
        int max_len = 0;
        for (const auto& [x, count] : freq) {
            auto it = freq.find(x + 1);
            if (it != freq.end()) {
                max_len = std::max(max_len, count + it->second);
            }
        }
        
        return max_len;
    }
};
```

#### Java 17
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public int findLHS(int[] nums) {
        Map<Integer, Integer> freq = new HashMap<>();
        for (int num : nums) {
            freq.put(num, freq.getOrDefault(num, 0) + 1);
        }
        
        int maxLen = 0;
        for (int x : freq.keySet()) {
            if (freq.containsKey(x + 1)) {
                maxLen = Math.max(maxLen, freq.get(x) + freq.get(x + 1));
            }
        }
        
        return maxLen;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - Building the frequency hash map takes $\mathcal{O}(n)$ time.
  - Iterating over the keys of the hash map and performing $\mathcal{O}(1)$ average lookups for $x + 1$ takes $\mathcal{O}(U)$ where $U \le n$ is the number of unique elements.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space
  - The hash map stores up to $n$ unique elements.

---

### Takeaway Pattern & Interview Traps

- **Subsequence vs. Subarray:** Always check whether the problem specifies *subarray* or *subsequence*. Subsequence means elements need not be contiguous, which frequently implies frequency counting or sorting will simplify the problem dramatically.
- **Requirement of Exactly 1:** $\max - \min$ must be **exactly** 1. An array of identical numbers like `[2, 2, 2]` has difference 0, so the answer is 0!