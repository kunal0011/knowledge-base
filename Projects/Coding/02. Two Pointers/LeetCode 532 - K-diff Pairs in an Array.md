---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 532: K-diff Pairs in an Array"
tags:
  - leetcode
  - coding
  - two-pointers
  - hash-map
  - sorting
  - amazon
  - google
---

# LeetCode 532: K-diff Pairs in an Array

**Target Companies:** Amazon, Google, Microsoft  
**Difficulty:** Medium  
**Topic:** Two Pointers / Value Deduplication / Sorted Difference Matching  

---

### Problem Statement

Given an array of integers `nums` and an integer `k`, return the number of **unique $k$-diff pairs** in the array.

A **$k$-diff pair** is defined as an integer pair `(nums[i], nums[j])`, where:
- $0 \le i, j < \text{nums.length}$
- $i \ne j$
- $|nums[i] - nums[j]| == k$

Notice that $|val|$ denotes the absolute value of $val$, and pairs $(x, y)$ and $(y, x)$ are counted as the same pair.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `k: int`
- **Output:** `int` (count of unique pairs)
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^4$
  - $-10^7 \le \text{nums}[i] \le 10^7$
  - $0 \le k \le 10^7$

---

### Key Idea & Intuition

The core objective is to identify **unique value pairs** $(u, v)$ such that $v - u = k$.

By sorting `nums` in ascending order:
1. Every valid pair satisfies $nums[right] - nums[left] = k$ with $left < right$.
2. We initialize two moving pointers: `left = 0` and `right = 1`.
3. If $nums[right] - nums[left] < k$, increase the gap by incrementing `right++`.
4. If $nums[right] - nums[left] > k$, decrease the gap by incrementing `left++`.
5. If $nums[right] - nums[left] == k$ (and $left \ne right$):
   - Increment our unique count `count++`.
   - Record the current values and fast-forward `left` past all duplicates of $nums[left]$ and `right` past all duplicates of $nums[right]$ to prevent double-counting.
6. Crucial invariant: $left < right$. Whenever $left == right$, immediately increment $right++$.

---

### Solution Approach (Step-by-Step)

1. Sort `nums` in ascending order.
2. Set `left = 0`, `right = 1`, `count = 0`.
3. While `right < len(nums)`:
   - If `left == right`: advance `right += 1` and continue.
   - Calculate `diff = nums[right] - nums[left]`.
   - If `diff < k`: advance `right += 1`.
   - Else if `diff > k`: advance `left += 1`.
   - Else (`diff == k`):
     - `count += 1`.
     - Store `left_val = nums[left]` and `right_val = nums[right]`.
     - Advance `left` while `left < len(nums)` and `nums[left] == left_val`.
     - Advance `right` while `right < len(nums)` and `nums[right] == right_val`.
4. Return `count`.

---

### Visual Algorithm Walkthrough

```
nums = [3, 1, 4, 1, 5], k = 2
Sorted: [1, 1, 3, 4, 5]
Index:   0  1  2  3  4

Pointers: left = 0 (1), right = 1 (1)
- diff = 1 - 1 = 0 < 2 -> right++ -> right = 2

Pointers: left = 0 (1), right = 2 (3)
- diff = 3 - 1 = 2 == k -> MATCH FOUND!
  Pair: (1, 3)
  count = 1
  Skip duplicate 1s: left moves past index 0, 1 -> left = 2
  Skip duplicate 3s: right moves past index 2 -> right = 3

Pointers: left = 2 (3), right = 3 (4)
- diff = 4 - 3 = 1 < 2 -> right++ -> right = 4

Pointers: left = 2 (3), right = 4 (5)
- diff = 5 - 3 = 2 == k -> MATCH FOUND!
  Pair: (3, 5)
  count = 2
  Skip duplicate 3s: left = 3
  Skip duplicate 5s: right = 5 (out of bounds)

Loop terminates.
Total Unique Pairs: 2 -> {(1, 3), (3, 5)}
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Positive K
- **Input:** `nums = [3, 1, 4, 1, 5]`, `k = 2`
- **Unique Pairs:** `(1, 3)` and `(3, 5)`.
- **Output:** `2`

#### Example 2: Zero Difference ($k = 0$)
- **Input:** `nums = [1, 3, 1, 5, 4]`, `k = 0`
- **Trace:** Only duplicate numbers qualify: $(1, 1)$.
- **Output:** `1`

#### Example 3: Multiple Identical Duplicates
- **Input:** `nums = [1, 2, 3, 4, 5]`, `k = 1`
- **Pairs:** `(1, 2)`, `(2, 3)`, `(3, 4)`, `(4, 5)`.
- **Output:** `4`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def findPairs(self, nums: List[int], k: int) -> int:
        if k < 0:
            return 0
            
        nums.sort()
        n = len(nums)
        left, right = 0, 1
        count = 0
        
        while right < n:
            if left == right:
                right += 1
                continue
                
            diff = nums[right] - nums[left]
            
            if diff < k:
                right += 1
            elif diff > k:
                left += 1
            else:
                count += 1
                left_val = nums[left]
                right_val = nums[right]
                
                # Advance pointers skipping duplicate values
                while left < n and nums[left] == left_val:
                    left += 1
                while right < n and nums[right] == right_val:
                    right += 1
                    
        return count
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int findPairs(std::vector<int>& nums, int k) {
        if (k < 0) return 0;
        
        std::sort(nums.begin(), nums.end());
        int n = nums.size();
        int left = 0, right = 1;
        int count = 0;
        
        while (right < n) {
            if (left == right) {
                right++;
                continue;
            }
            
            int diff = nums[right] - nums[left];
            if (diff < k) {
                right++;
            } else if (diff > k) {
                left++;
            } else {
                count++;
                int leftVal = nums[left];
                int rightVal = nums[right];
                
                while (left < n && nums[left] == leftVal) left++;
                while (right < n && nums[right] == rightVal) right++;
            }
        }
        
        return count;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public int findPairs(int[] nums, int k) {
        if (k < 0) return 0;

        Arrays.sort(nums);
        int n = nums.length;
        int left = 0, right = 1;
        int count = 0;

        while (right < n) {
            if (left == right) {
                right++;
                continue;
            }

            int diff = nums[right] - nums[left];
            if (diff < k) {
                right++;
            } else if (diff > k) {
                left++;
            } else {
                count++;
                int leftVal = nums[left];
                int rightVal = nums[right];

                while (left < n && nums[left] == leftVal) left++;
                while (right < n && nums[right] == rightVal) right++;
            }
        }

        return count;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N \log N)$ where $N = \text{len(nums)}$ due to the initial sort. The two-pointer traversal takes $O(N)$ total time as each pointer increments at most $N$ times.
- **Space Complexity:** $O(1)$ auxiliary space if sorting in-place.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Same-Direction Dual Pointer on Sorted Arrays with Duplicate Skipping.
- **Trap:** `k == 0` edge case: If $k = 0$, $nums[left] == nums[right]$. If $left == right$ is not guarded, the algorithm would incorrectly compare an element with itself ($diff = 0$) and increment count. Enforcing $left < right$ prevents this.