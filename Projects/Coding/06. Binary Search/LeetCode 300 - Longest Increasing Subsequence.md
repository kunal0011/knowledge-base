---
date: "2026-09-15"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 300: Longest Increasing Subsequence"
tags:
  - leetcode
  - coding
  - binary-search
  - dynamic-programming
  - patience-sorting
  - google
  - amazon
---

# LeetCode 300: Longest Increasing Subsequence

**Target Companies:** Google (All-Time Classic), Amazon, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Patience Sorting via Binary Search ($O(N \log N)$ LIS)  

---

### Problem Statement

Given an integer array `nums`, return the length of the **longest strictly increasing subsequence**.

A **subsequence** is an array that can be derived from another array by deleting some or no elements without changing the order of the remaining elements.

Follow up: Can you come up with an algorithm that runs in $O(n \log n)$ time complexity?

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `int` (length of longest increasing subsequence)
- **Constraints:**
  - $1 \le \text{nums.length} \le 2500$
  - $-10^4 \le \text{nums}[i] \le 10^4$

---

### Key Idea & Intuition

#### The $O(N^2)$ Dynamic Programming Baseline:
Let $dp[i]$ be the length of the LIS ending at index $i$.
$$dp[i] = 1 + \max_{0 \le j < i, nums[j] < nums[i]} (dp[j])$$
This takes $O(N^2)$ time, which is acceptable for $N \le 2500$, but suboptimal.

#### The $O(N \log N)$ Patience Sorting Technique:
Consider the Solitaire card game **Patience Sorting**:
- We maintain an array `tails`, where `tails[i]` stores the **smallest ending element** of all increasing subsequences of length $i + 1$ found so far.
- **Key Invariant:** The array `tails` is **strictly monotonically increasing**.
  - Why? A subsequence of length $k + 1$ requires an element strictly greater than the tail of a valid subsequence of length $k$.
- Because `tails` is sorted, for each incoming number $x \in nums$, we can find its insertion position using **Binary Search** (`bisect_left`):
  1. If $x$ is greater than all values in `tails`:
     - $x$ extends the longest known subsequence! Append $x$ to `tails`.
  2. Else, find the first element in `tails` such that `tails[idx] >= x`:
     - Replace `tails[idx] = x`.
     - *Why replace?* A smaller tail element makes it easier for future numbers to extend this subsequence!
- At the end, the length of `tails` is the exact length of the Longest Increasing Subsequence.

---

### Solution Approach (Step-by-Step)

1. Initialize an empty list `tails = []`.
2. For each number $x$ in `nums`:
   - Perform binary search to find the smallest index `idx` such that `tails[idx] >= x` (Lower Bound).
   - If `idx == len(tails)`:
     - `tails.append(x)`
   - Else:
     - `tails[idx] = x`
3. Return `len(tails)`.

---

### Visual Algorithm Walkthrough

```
nums = [10, 9, 2, 5, 3, 7, 101, 18]

x = 10:  tails = [10]
x = 9:   9 < 10 -> replace 10 -> tails = [9]
x = 2:   2 < 9  -> replace 9  -> tails = [2]
x = 5:   5 > 2  -> append 5   -> tails = [2, 5]
x = 3:   2 < 3 <= 5 -> replace 5 -> tails = [2, 3]
x = 7:   7 > 3  -> append 7   -> tails = [2, 3, 7]
x = 101: 101 > 7 -> append 101 -> tails = [2, 3, 7, 101]
x = 18:  7 < 18 <= 101 -> replace 101 -> tails = [2, 3, 7, 18]

Length of tails = 4.
LIS length = 4 (e.g., [2, 3, 7, 18] or [2, 5, 7, 101]).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Mixed Sequence
- **Input:** `nums = [10, 9, 2, 5, 3, 7, 101, 18]`
- **Tails Evolution:** `[10] -> [9] -> [2] -> [2, 5] -> [2, 3] -> [2, 3, 7] -> [2, 3, 7, 101] -> [2, 3, 7, 18]`
- **Output:** `4`

#### Example 2: Non-Decreasing Sequence with Duplicates
- **Input:** `nums = [0, 1, 0, 3, 2, 3]`
- **Output:** `4` (Subsequence `[0, 1, 2, 3]`)

#### Example 3: All Identical Elements
- **Input:** `nums = [7, 7, 7, 7, 7]`
- **Trace:** Since strictly increasing is required, `tails` stays `[7]`.
- **Output:** `1`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
import bisect

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        tails: List[int] = []
        
        for x in nums:
            # Find the leftmost index where tails[idx] >= x
            idx = bisect.bisect_left(tails, x)
            if idx == len(tails):
                tails.append(x)
            else:
                tails[idx] = x
                
        return len(tails)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int lengthOfLIS(std::vector<int>& nums) {
        std::vector<int> tails;

        for (int x : nums) {
            // std::lower_bound returns iterator to first element >= x
            auto it = std::lower_bound(tails.begin(), tails.end(), x);
            if (it == tails.end()) {
                tails.push_back(x);
            } else {
                *it = x;
            }
        }

        return static_cast<int>(tails.size());
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class Solution {
    public int lengthOfLIS(int[] nums) {
        List<Integer> tails = new ArrayList<>();

        for (int x : nums) {
            int idx = Collections.binarySearch(tails, x);
            if (idx < 0) {
                // If not found, binarySearch returns (-(insertion point) - 1)
                idx = -(idx + 1);
            }

            if (idx == tails.size()) {
                tails.add(x);
            } else {
                tails.set(idx, x);
            }
        }

        return tails.size();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N \log N)$ — For each of the $N$ numbers, we perform a binary search over `tails` of length at most $N$. Each lookup takes $O(\log N)$.
- **Space Complexity:** $O(N)$ auxiliary space — To store the `tails` array of size at most $N$.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Patience Sorting + Binary Search for Optimal Subsequence Length.
- **Trap:** `tails` contains the **length** of the LIS, but NOT necessarily the actual elements of the LIS in order! The elements in `tails` represent the minimal tails of subsequences of various lengths and may be updated out-of-order relative to the original sequence.