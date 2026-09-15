---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 217: Contains Duplicate"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - hash-table
  - amazon
  - apple
  - google
  - meta
---

# LeetCode 217: Contains Duplicate

**Target Companies:** Amazon, Apple, Google, Microsoft, Meta  
**Difficulty:** Easy  
**Topic:** Hash Set Membership / Early Exit

---

### Problem Statement

Given an integer array `nums`, return `true` if any value appears **at least twice** in the array, and return `false` if every element is distinct.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `bool`
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $-10^9 \le \text{nums}[i] \le 10^9$

---

### Key Idea & Intuition

The problem tests detection of duplicate elements across an unsorted sequence:

1. **Brute Force:**
   - Comparing each pair $(i, j)$ takes $\mathcal{O}(n^2)$ time, which TLEs for $N = 10^5$.
2. **Sorting Approach:**
   - Sorting takes $\mathcal{O}(n \log n)$ time and $\mathcal{O}(1)$ extra space. Adjacent elements can be checked for equality.
3. **Hash Set Approach (Optimal Time):**
   - Maintain a hash set `seen` of elements visited so far.
   - For each number `num` in `nums`:
     - If `num in seen`, a duplicate is found. Return `True` immediately (early exit).
     - Otherwise, insert `num` into `seen`.
   - If the loop finishes without finding any duplicate, return `False`.
   - In Python, comparing `len(nums) != len(set(nums))` is also a concise one-liner, though early exit with a loop is more efficient for large inputs where duplicates appear early.

---

### Solution Approach (Step-by-Step)

1. **Initialize Hash Set:**
   - Create an empty hash set `seen`.
2. **Single Pass with Early Exit:**
   - For each number `num` in `nums`:
     - If `num in seen`, return `True`.
     - Add `num` to `seen`.
3. **Fallback:**
   - If loop terminates, all elements are unique. Return `False`.

---

### Visual Algorithm Walkthrough

#### Example: `nums = [1, 2, 3, 1]`

```
Step 0: seen = {}

Index 0, num = 1:
  1 in seen? No.
  seen.add(1) -> seen = {1}

Index 1, num = 2:
  2 in seen? No.
  seen.add(2) -> seen = {1, 2}

Index 2, num = 3:
  3 in seen? No.
  seen.add(3) -> seen = {1, 2, 3}

Index 3, num = 1:
  1 in seen? YES! Duplicate detected!
  Return True immediately.
```

---

### Solved Examples with Multiple Inputs

| Input `nums` | Elements Added to Set | Duplicate Encountered? | Output |
| :--- | :--- | :--- | :--- |
| `[1, 2, 3, 1]` | `1, 2, 3` $\to$ `1` seen again | Yes | `True` |
| `[1, 2, 3, 4]` | `1, 2, 3, 4` | No | `False` |
| `[1, 1, 1, 3, 3, 4, 3, 2, 4, 2]` | `1` $\to$ `1` seen at index 1 | Yes (Immediate exit) | `True` |
| `[42]` | `42` | No | `False` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_set>

class Solution {
public:
    bool containsDuplicate(std::vector<int>& nums) {
        std::unordered_set<int> seen;
        seen.reserve(nums.size());
        
        for (int num : nums) {
            if (seen.count(num)) {
                return true;
            }
            seen.insert(num);
        }
        return false;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.HashSet;
import java.util.Set;

class Solution {
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> seen = new HashSet<>();
        for (int num : nums) {
            if (!seen.add(num)) {
                return true;
            }
        }
        return false;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$ in the worst case (when all elements are unique or duplicate is at the very end). On average, early exit occurs faster than examining all $n$ items.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space to store up to $n$ unique elements in the hash set.

---

### Takeaway Pattern & Interview Traps

1. **Tradeoff Between Time and Space:**
   - Hash Set achieves $\mathcal{O}(n)$ time and $\mathcal{O}(n)$ space.
   - In-place sorting achieves $\mathcal{O}(n \log n)$ time and $\mathcal{O}(1)$ extra space.
2. **Java `Set.add()` Return Value Trick:**
   - In Java, `Set.add(e)` returns `false` if the element was already present in the set. Checking `if (!seen.add(num)) return true;` does lookup and insertion in a single step.
3. **C++ `reserve()` Optimization:**
   - Pre-allocating buckets with `seen.reserve(nums.size())` eliminates hash table rehashing overhead during insertion.
