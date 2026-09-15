---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 41: First Missing Positive"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - amazon
  - google
---

# LeetCode 41: First Missing Positive

**Target Companies:** Google (Signature Hard), Amazon, Meta, Microsoft  
**Difficulty:** Hard  
**Topic:** In-Place Cyclic Sort / Array as Hash Map

---

### Problem Statement

Given an unsorted integer array `nums`, return the **smallest missing positive integer**.

You must implement an algorithm that runs in **$O(n)$ time** and uses **$O(1)$ auxiliary space**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `int` (smallest positive integer not in nums)
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $-2^{31} \le \text{nums}[i] \le 2^{31} - 1$

---

### Key Idea & Intuition

- **Pigeonhole Principle:**
  - For an array of length $N$, the smallest missing positive integer MUST lie in the range $[1, N + 1]$.
  - If all numbers $1, 2, \dots, N$ are present, the answer is $N + 1$.
  - Otherwise, the answer is the first number in $1, \dots, N$ that is absent.
- **Cyclic Placement (In-Place Index Mapping):**
  - Place every number $x \in [1, N]$ into its correct home index $x - 1$:
    `nums[nums[i] - 1] == nums[i]`.
  - While $1 \le \text{nums}[i] \le N$ and $\text{nums}[i] \ne \text{nums}[\text{nums}[i] - 1]$:
    - Swap `nums[i]` with `nums[nums[i] - 1]`.
  - Second pass: the first index $i$ where $\text{nums}[i] \ne i + 1$ reveals the missing positive $i + 1$!

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        
        for i in range(n):
            while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
                correct_idx = nums[i] - 1
                nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
                
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
                
        return n + 1
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int firstMissingPositive(std::vector<int>& nums) {
        int n = nums.size();

        for (int i = 0; i < n; ++i) {
            while (nums[i] >= 1 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                std::swap(nums[i], nums[nums[i] - 1]);
            }
        }

        for (int i = 0; i < n; ++i) {
            if (nums[i] != i + 1) return i + 1;
        }

        return n + 1;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;

        for (int i = 0; i < n; i++) {
            while (nums[i] >= 1 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                int correctIdx = nums[i] - 1;
                int temp = nums[i];
                nums[i] = nums[correctIdx];
                nums[correctIdx] = temp;
            }
        }

        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) return i + 1;
        }

        return n + 1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Each swap places at least one number into its permanent correct position; no number is moved more than twice.
- **Space Complexity:** $O(1)$ strict auxiliary in-place space.
