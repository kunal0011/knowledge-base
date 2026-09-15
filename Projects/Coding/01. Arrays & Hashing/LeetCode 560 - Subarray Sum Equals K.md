---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 560: Subarray Sum Equals K"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - prefix-sum
  - hash-table
  - meta
  - google
  - amazon
---

# LeetCode 560: Subarray Sum Equals K

**Target Companies:** Meta (Top #1 Classic), Google, Amazon, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Prefix Sum + Hash Map Frequency Counting

---

### Problem Statement

Given an array of integers `nums` and an integer `k`, return *the total number of subarrays whose sum equals to `k`*.

A subarray is a contiguous non-empty sequence of elements within an array.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `k: int`
- **Output:** `int`
- **Constraints:**
  - $1 \le \text{nums.length} \le 2 \times 10^4$
  - $-1000 \le \text{nums}[i] \le 1000$
  - $-10^7 \le k \le 10^7$

---

### Key Idea & Intuition

#### 1. Why Sliding Window Fails
A standard two-pointer sliding window assumes that expanding the right pointer monotonically increases the window sum, and shrinking the left pointer monotonically decreases it. Because `nums` can contain **negative integers**, this monotonicity does not hold. A negative number can decrease the sum when expanding or increase it when shrinking.

#### 2. Prefix Sum Transformation
Let $P[i]$ be the prefix sum of elements from index $0$ to $i$:
$$P[i] = \sum_{m=0}^i \text{nums}[m]$$
The sum of any contiguous subarray $\text{nums}[j \dots i]$ ($j \le i$) is:
$$\text{sum}(\text{nums}[j \dots i]) = P[i] - P[j - 1]$$
We seek the number of pairs $(j, i)$ such that:
$$P[i] - P[j - 1] = k \iff P[j - 1] = P[i] - k$$

#### 3. Hash Map of Prefix Sum Frequencies
As we compute the running prefix sum $P[i]$:
- How many earlier prefix sums equal $P[i] - k$?
- By maintaining a frequency map `prefix_counts` that stores how many times each prefix sum has occurred so far, we simply add `prefix_counts[P[i] - k]` to our running answer in $\mathcal{O}(1)$ time!
- **Base Case:** Initialize `prefix_counts[0] = 1`. This represents the empty prefix before index 0, allowing subarrays starting from index 0 whose sum is $k$ to be counted when $P[i] - k = 0$.

---

### Solution Approach (Step-by-Step)

1. **Initialize Data Structures:**
   - Create a hash map `prefix_counts` initialized with `{0: 1}`.
   - Maintain `curr_sum = 0` and `total_subarrays = 0`.
2. **Iterate Through Elements:**
   - For each number `num` in `nums`:
     - Add `num` to `curr_sum`.
     - Target earlier prefix is `target = curr_sum - k`.
     - Add `prefix_counts.get(target, 0)` to `total_subarrays`.
     - Increment the frequency of `curr_sum` in `prefix_counts`: `prefix_counts[curr_sum] += 1`.
3. **Return Result:**
   - Return `total_subarrays`.

---

### Visual Algorithm Walkthrough

#### Example: `nums = [1, 2, 3, -2, 2]`, `k = 3`

```
Initial: prefix_counts = {0: 1}, curr_sum = 0, total = 0

i=0, num=1:
  curr_sum = 1
  target = 1 - 3 = -2 (in map? No, 0)
  map[1] = 1 -> {0: 1, 1: 1}

i=1, num=2:
  curr_sum = 3
  target = 3 - 3 = 0 (in map? Yes, count=1) -> Subarray [1, 2]
  total += 1 = 1
  map[3] = 1 -> {0: 1, 1: 1, 3: 1}

i=2, num=3:
  curr_sum = 6
  target = 6 - 3 = 3 (in map? Yes, count=1) -> Subarray [3]
  total += 1 = 2
  map[6] = 1 -> {0: 1, 1: 1, 3: 1, 6: 1}

i=3, num=-2:
  curr_sum = 4
  target = 4 - 3 = 1 (in map? Yes, count=1) -> Subarray [2, 3, -2] (sum = 3)
  total += 1 = 3
  map[4] = 1

i=4, num=2:
  curr_sum = 6
  target = 6 - 3 = 3 (in map? Yes, count=1) -> Subarray [-2, 2, 3...]
  total += 1 = 4
  map[6] = 2

Final total = 4 subarrays.
```

---

### Solved Examples with Multiple Inputs

| Input `nums` | `k` | Prefix Sums Generated | Valid Subarrays Found | Output |
| :--- | :--- | :--- | :--- | :--- |
| `[1, 1, 1]` | `2` | `[1, 2, 3]` | `[1, 1]` at index 0-1, 1-2 | `2` |
| `[1, 2, 3]` | `3` | `[1, 3, 6]` | `[1, 2]` at 0-1, `[3]` at 2 | `2` |
| `[1, -1, 0]` | `0` | `[1, 0, 0]` | `[1, -1]`, `[0]`, `[1, -1, 0]` | `3` |
| `[-1, -1, 1]` | `0` | `[-1, -2, -1]` | `[-1, 1]` at 1-2 | `1` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import defaultdict

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        prefix_counts = defaultdict(int)
        prefix_counts[0] = 1
        
        curr_sum = 0
        total_subarrays = 0
        
        for num in nums:
            curr_sum += num
            total_subarrays += prefix_counts[curr_sum - k]
            prefix_counts[curr_sum] += 1
            
        return total_subarrays
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_map>

class Solution {
public:
    int subarraySum(std::vector<int>& nums, int k) {
        std::unordered_map<int, int> prefixCounts;
        prefixCounts[0] = 1;

        int currSum = 0;
        int total = 0;

        for (int num : nums) {
            currSum += num;
            auto it = prefixCounts.find(currSum - k);
            if (it != prefixCounts.end()) {
                total += it->second;
            }
            prefixCounts[currSum]++;
        }

        return total;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> prefixCounts = new HashMap<>();
        prefixCounts.put(0, 1);

        int currSum = 0;
        int total = 0;

        for (int num : nums) {
            currSum += num;
            total += prefixCounts.getOrDefault(currSum - k, 0);
            prefixCounts.put(currSum, prefixCounts.getOrDefault(currSum, 0) + 1);
        }

        return total;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$, where $n$ is the length of `nums`. We perform a single linear scan through the array with $\mathcal{O}(1)$ average hash table insertions and lookups.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space to store at most $n + 1$ unique prefix sums in the frequency map.

---

### Takeaway Pattern & Interview Traps

1. **The Crucial `prefix_counts[0] = 1` Base Case:**
   - If `nums = [3]` and `k = 3`, then at index 0, `curr_sum = 3`. We look for `curr_sum - k = 0`. If `{0: 1}` was not pre-populated, this valid single-element subarray would be missed!
2. **Lookup Before Insertion Order:**
   - Always query for `curr_sum - k` **before** incrementing `prefix_counts[curr_sum]`. If $k = 0$, doing the reverse would cause a prefix sum to pair with itself, incorrectly counting an empty subarray.
3. **Difference from Two Sum:**
   - Two Sum looks for pairs of indices; Subarray Sum looks for ranges via prefix differences. Both reduce $\mathcal{O}(n^2)$ pair searches to linear time via hash maps.
