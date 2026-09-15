---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 1: Two Sum"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - amazon
  - google
---

# LeetCode 1: Two Sum

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Apple, Microsoft  
**Difficulty:** Easy  
**Topic:** Hash Map Lookup / One-Pass Complement

---

### Problem Statement

Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

You may assume that each input would have **exactly one solution**, and you may not use the same element twice.

You can return the answer in any order.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `target: int`
- **Output:** `List[int]` of length 2 containing the 0-based indices.
- **Constraints:**
  - $2 \le \text{nums.length} \le 10^4$
  - $-10^9 \le \text{nums}[i] \le 10^9$
  - $-10^9 \le \text{target} \le 10^9$
  - Only one valid answer exists.

---

### Key Idea & Intuition

- **Brute Force Failure:**
  - Checking every pair `(i, j)` takes $O(N^2)$ time.
- **The Hash Map Complement Principle:**
  - For any element $x = \text{nums}[i]$, its required pair is strictly defined as $\text{complement} = \text{target} - x$.
  - Instead of looking forward in $O(N)$, we maintain a hash map of all previously visited numbers: `num -> index`.
  - On encountering each number, we query the hash map for `target - x` in $O(1)$ average time.
  - If found, we immediately return `[seen[target - x], i]`.

---

### Solution Approach (Step-by-Step)

1. Initialize an empty hash map `seen = {}`.
2. Iterate `i, num` through `enumerate(nums)`:
   - Calculate `complement = target - num`.
   - If `complement in seen`:
     - Return `[seen[complement], i]`.
   - Otherwise, record `seen[num] = i`.
3. Return `[]` (fallback).

---

### Visual Algorithm Walkthrough

```
nums = [2, 7, 11, 15], target = 9

i=0, num=2:
  complement = 9 - 2 = 7
  7 in seen? NO
  seen = {2: 0}

i=1, num=7:
  complement = 9 - 7 = 2
  2 in seen? YES! -> index 0
  Match found: [0, 1]

Result: [0, 1]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Positive Array
- **Input:** `nums = [2, 7, 11, 15], target = 9`
- **Output:** `[0, 1]`

#### Example 2: Negative Numbers
- **Input:** `nums = [-3, 4, 3, 90], target = 0`
- **Trace:**
  - `i=0, num=-3`: comp = 3 -> `seen = {-3: 0}`
  - `i=1, num=4`: comp = -4 -> `seen = {-3: 0, 4: 1}`
  - `i=2, num=3`: comp = -3 -> found at index 0! Return `[0, 2]`.
- **Output:** `[0, 2]`

#### Example 3: Identical Elements
- **Input:** `nums = [3, 3], target = 6`
- **Output:** `[0, 1]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i
        return []
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_map>

class Solution {
public:
    std::vector<int> twoSum(std::vector<int>& nums, int target) {
        std::unordered_map<int, int> seen;
        for (int i = 0; i < nums.size(); ++i) {
            int complement = target - nums[i];
            if (seen.count(complement)) {
                return {seen[complement], i};
            }
            seen[nums[i]] = i;
        }
        return {};
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> seen = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (seen.containsKey(complement)) {
                return new int[] { seen.get(complement), i };
            }
            seen.put(nums[i], i);
        }
        return new int[0];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — One pass through the array. Hash map insertions and lookups take average $O(1)$ time.
- **Space Complexity:** $O(N)$ — Stores up to $N - 1$ key-value pairs in the hash map.
