---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 1574: Shortest Subarray to be Removed to Make Array Sorted"
tags:
  - leetcode
  - coding
  - two-pointers
  - prefix-suffix
  - amazon
  - google
---

# LeetCode 1574: Shortest Subarray to be Removed to Make Array Sorted

**Target Companies:** Google, Amazon, Bloomberg  
**Difficulty:** Medium  
**Topic:** Two Pointers / Non-Decreasing Prefix-Suffix Matching  

---

### Problem Statement

Given an integer array `arr`, remove **one contiguous subarray** (possibly empty) such that the remaining elements are **non-decreasing** (i.e., `arr[i] <= arr[i + 1]` for all valid $i$).

Return the **minimum length** of the subarray that must be removed.

---

### Input & Output Formats & Constraints

- **Input:** `arr: List[int]`
- **Output:** `int` (minimum length of the contiguous subarray to remove)
- **Constraints:**
  - $1 \le \text{arr.length} \le 10^5$
  - $0 \le \text{arr}[i] \le 10^9$

---

### Key Idea & Intuition

Any valid removal leaves:
1. A non-decreasing **prefix** `arr[0 ... left]`.
2. A non-decreasing **suffix** `arr[right ... n - 1]`.
3. The removed segment is strictly between some prefix element $i$ and suffix element $j$ such that `arr[i] <= arr[j]`.

Notice three initial baseline choices:
- Keep only the longest non-decreasing prefix `arr[0 ... left]`: remove length is $n - 1 - left$.
- Keep only the longest non-decreasing suffix `arr[right ... n - 1]`: remove length is $right$.
- If `left >= right`, the array is already sorted: remove length is $0$.

To improve further, we merge both sides: use two pointers `i = 0` (on prefix) and `j = right` (on suffix). If `arr[i] <= arr[j]`, elements between `i` and `j` (exclusive) can be removed, giving a candidate removal length of $j - i - 1$. Then we advance $i++$ to attempt keeping a larger prefix. If `arr[i] > arr[j]`, we must advance $j++$ until $arr[j] \ge arr[i]$ or $j = n$.

---

### Solution Approach (Step-by-Step)

1. Find the longest non-decreasing prefix: advance `left` while `arr[left] <= arr[left + 1]`.
2. If `left == n - 1`, array is already sorted, return `0`.
3. Find the longest non-decreasing suffix: decrement `right` starting from `n - 1` while `arr[right - 1] <= arr[right]`.
4. Initialize answer with the single-sided removals: `ans = min(n - left - 1, right)`.
5. Two Pointers traversal:
   - Start `i = 0`, `j = right`.
   - While `i <= left` and `j < n`:
     - If `arr[i] <= arr[j]`:
       - `ans = min(ans, j - i - 1)`
       - `i += 1`
     - Else:
       - `j += 1`
6. Return `ans`.

---

### Visual Algorithm Walkthrough

```
Input: arr = [1, 2, 3, 10, 4, 2, 3, 5]
              -------  ------  -------
              Prefix   Middle  Suffix

1. Identify Prefix:
   [1, 2, 3, 10]  -> left = 3 (since 10 > 4)

2. Identify Suffix:
   [2, 3, 5]      -> right = 5 (since 4 > 2, arr[5]=2 <= arr[6]=3 <= arr[7]=5)

3. Baseline Options:
   Remove suffix: remove arr[4..7] -> length 8 - 3 - 1 = 4
   Remove prefix: remove arr[0..4] -> length 5
   ans = min(4, 5) = 4

4. Two-Pointer Merging (i in [0..3], j in [5..7]):
   i=0 (1), j=5 (2): arr[0] <= arr[5] (1 <= 2)
     -> Valid! Remove arr[1..4] -> length j - i - 1 = 5 - 0 - 1 = 4. i++ -> 1.
   i=1 (2), j=5 (2): arr[1] <= arr[5] (2 <= 2)
     -> Valid! Remove arr[2..4] -> length j - i - 1 = 5 - 1 - 1 = 3. ans = min(4, 3) = 3. i++ -> 2.
   i=2 (3), j=5 (2): arr[2] > arr[5] (3 > 2)
     -> Invalid! Must increase j. j++ -> 6.
   i=2 (3), j=6 (3): arr[2] <= arr[6] (3 <= 3)
     -> Valid! Remove arr[3..5] -> length 6 - 2 - 1 = 3. ans = 3. i++ -> 3.
   i=3 (10), j=6 (3): arr[3] > arr[6] (10 > 3)
     -> Invalid! j++ -> 7.
   i=3 (10), j=7 (5): arr[3] > arr[7] (10 > 5)
     -> Invalid! j++ -> 8 (out of bounds).

End of Loop. Minimum subarray length to remove = 3. (Remove [10, 4, 2])
Remaining array: [1, 2, 3, 3, 5] (Sorted!).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Mixed Input
- **Input:** `arr = [1, 2, 3, 10, 4, 2, 3, 5]`
- **Prefix:** `[1, 2, 3, 10]` (`left = 3`)
- **Suffix:** `[2, 3, 5]` (`right = 5`)
- **Optimal Cut:** Remove `[10, 4, 2]` (length 3).
- **Output:** `3`

#### Example 2: Already Non-Decreasing
- **Input:** `arr = [1, 2, 3, 4, 5]`
- **Prefix:** `left = 4` (`left == n - 1`)
- **Output:** `0`

#### Example 3: Strictly Decreasing
- **Input:** `arr = [5, 4, 3, 2, 1]`
- **Prefix:** `left = 0` (`arr[0] = 5`)
- **Suffix:** `right = 4` (`arr[4] = 1`)
- **Baseline:** `ans = min(5 - 0 - 1, 4) = 4`
- **Output:** `4` (keep any single element, remove the other 4).

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def findLengthOfShortestSubarray(self, arr: List[int]) -> int:
        n = len(arr)
        
        # 1. Find non-decreasing prefix boundary
        left = 0
        while left + 1 < n and arr[left] <= arr[left + 1]:
            left += 1
            
        if left == n - 1:
            return 0
            
        # 2. Find non-decreasing suffix boundary
        right = n - 1
        while right > 0 and arr[right - 1] <= arr[right]:
            right -= 1
            
        # 3. Base answer: remove either entire suffix or entire prefix
        ans = min(n - left - 1, right)
        
        # 4. Merge prefix [0..left] and suffix [right..n-1]
        i, j = 0, right
        while i <= left and j < n:
            if arr[i] <= arr[j]:
                ans = min(ans, j - i - 1)
                i += 1
            else:
                j += 1
                
        return ans
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int findLengthOfShortestSubarray(std::vector<int>& arr) {
        int n = arr.size();
        
        // 1. Longest non-decreasing prefix
        int left = 0;
        while (left + 1 < n && arr[left] <= arr[left + 1]) {
            left++;
        }
        if (left == n - 1) return 0;
        
        // 2. Longest non-decreasing suffix
        int right = n - 1;
        while (right > 0 && arr[right - 1] <= arr[right]) {
            right--;
        }
        
        // 3. Baseline answers
        int ans = std::min(n - left - 1, right);
        
        // 4. Two-pointer merge
        int i = 0, j = right;
        while (i <= left && j < n) {
            if (arr[i] <= arr[j]) {
                ans = std::min(ans, j - i - 1);
                i++;
            } else {
                j++;
            }
        }
        
        return ans;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int findLengthOfShortestSubarray(int[] arr) {
        int n = arr.length;
        
        // 1. Longest non-decreasing prefix
        int left = 0;
        while (left + 1 < n && arr[left] <= arr[left + 1]) {
            left++;
        }
        if (left == n - 1) return 0;
        
        // 2. Longest non-decreasing suffix
        int right = n - 1;
        while (right > 0 && arr[right - 1] <= arr[right]) {
            right--;
        }
        
        // 3. Initial baseline
        int ans = Math.min(n - left - 1, right);
        
        // 4. Two pointers merging
        int i = 0, j = right;
        while (i <= left && j < n) {
            if (arr[i] <= arr[j]) {
                ans = Math.min(ans, j - i - 1);
                i++;
            } else {
                j++;
            }
        }
        
        return ans;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Finding `left` and `right` takes at most $O(N)$. The two pointers `i` and `j` advance monotonically, each traversing at most $N$ positions. Overall time is strictly linear.
- **Space Complexity:** $O(1)$ — Only scalar index variables are maintained.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Prefix-Suffix Decomposition with Monotonic Two-Pointer Stitching. When allowed to eliminate a single contiguous subarray to satisfy a sorted/validity property, always look at the valid prefix and suffix boundaries.
- **Trap:** Forgetting that removing only prefix elements (`arr[0...right-1]`) or only suffix elements (`arr[left+1...n-1]`) are completely valid candidates without needing to match any elements.