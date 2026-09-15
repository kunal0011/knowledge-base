---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 1403: Minimum Subsequence in Non-Increasing Order"
tags:
  - leetcode
  - coding
  - greedy
  - sorting
  - counting-sort
  - amazon
  - google
---

# LeetCode 1403: Minimum Subsequence in Non-Increasing Order

**Target Companies:** Amazon, Google, Microsoft  
**Difficulty:** Easy  
**Topic:** Greedy / Sorting / Counting Sort  

---

### Problem Statement

Given the array `nums`, obtain a subsequence of the array whose sum of elements is **strictly greater** than the sum of the non-included elements in such a way that:
1. The number of elements in the subsequence is **minimized**.
2. If there are multiple solutions, return the subsequence with the **maximum total sum** of its elements.
3. The returned subsequence must be in **non-increasing order** (sorted in descending order).

A **subsequence** of an array can be obtained by erasing some (possibly zero) elements from the array.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{nums.length} \le 500$).
- **Output:**
  - `List[int]` / `vector<int>` / `int[]` — the chosen subsequence in descending order.
- **Constraints:**
  - $1 \le \text{nums.length} \le 500$
  - $1 \le \text{nums}[i] \le 100$

---

### Key Idea & Intuition

Let $S = \sum_{x \in \text{nums}} x$ be the total sum of the array.
We need to pick a subset of elements with sum $S_{\text{sub}}$ such that:
$$S_{\text{sub}} > S - S_{\text{sub}} \iff 2 \cdot S_{\text{sub}} > S \iff S_{\text{sub}} > \frac{S}{2}$$

To satisfy both objectives:
1. **Minimize the number of elements selected:** Each selection should contribute as much value as possible toward exceeding $S / 2$.
2. **Maximize the sum of the chosen elements:** Selecting the largest elements first naturally maximizes the total sum for any fixed subset size.

Therefore, the **Greedy Choice** is to sort the numbers in descending order and greedily accumulate elements from largest to smallest until the running sum strictly exceeds half of the total array sum.
Since elements are sorted descending, the output is automatically in non-increasing order.

---

### Solution Approach (Step-by-Step)

1. Compute `total_sum = sum(nums)` and `target = total_sum // 2`.
2. Sort `nums` in descending order.
3. Initialize `subseq = []` and `curr_sum = 0`.
4. Iterate through the sorted `nums`:
   - Append `num` to `subseq`.
   - Increment `curr_sum += num`.
   - If `curr_sum > target`:
     - Break and return `subseq`.
5. Return `subseq`.

---

### Visual Algorithm Walkthrough

For `nums = [4, 3, 10, 9, 8]`:

```
Total sum S = 4 + 3 + 10 + 9 + 8 = 34
Half sum = 34 // 2 = 17
Requirement: curr_sum > 17

1. Sort descending: [10, 9, 8, 4, 3]

2. Greedily pick from left:
   - Pick 10: curr_sum = 10 <= 17. subseq = [10]
   - Pick 9:  curr_sum = 10 + 9 = 19 > 17 (STRICTLY GREATER!)
     Remaining sum: 34 - 19 = 15.
     19 > 15 -> CONDITION MET!

Stop.
Result = [10, 9].
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [4, 3, 10, 9, 8]`
- **Output:** `[10, 9]`

#### Example 2:
- **Input:** `nums = [4, 4, 7, 6, 7]`
- **Tracing:**
  - $S = 28$, half = 14.
  - Sorted: `[7, 7, 6, 4, 4]`.
  - Pick 7: sum = 7.
  - Pick 7: sum = 14 (not strictly greater).
  - Pick 6: sum = 20 > 14.
- **Output:** `[7, 7, 6]`

#### Example 3 (Single Element):
- **Input:** `nums = [6]`
- **Output:** `[6]`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def minSubsequence(self, nums: List[int]) -> List[int]:
        nums.sort(reverse=True)
        half_sum = sum(nums) // 2
        
        curr_sum = 0
        result: List[int] = []
        
        for x in nums:
            curr_sum += x
            result.append(x)
            if curr_sum > half_sum:
                break
                
        return result
```

#### C++17
```cpp
#include <vector>
#include <numeric>
#include <algorithm>

class Solution {
public:
    std::vector<int> minSubsequence(std::vector<int>& nums) {
        std::sort(nums.rbegin(), nums.rend());
        int total_sum = std::accumulate(nums.begin(), nums.end(), 0);
        int half_sum = total_sum / 2;
        
        int curr_sum = 0;
        std::vector<int> result;
        
        for (int x : nums) {
            curr_sum += x;
            result.push_back(x);
            if (curr_sum > half_sum) {
                break;
            }
        }
        
        return result;
    }
};
```

#### Java 17
```java
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class Solution {
    public List<Integer> minSubsequence(int[] nums) {
        Arrays.sort(nums);
        int totalSum = 0;
        for (int num : nums) {
            totalSum += num;
        }
        int halfSum = totalSum / 2;
        
        List<Integer> result = new ArrayList<>();
        int currSum = 0;
        
        // Traverse from right to left (largest to smallest)
        for (int i = nums.length - 1; i >= 0; i--) {
            currSum += nums[i];
            result.add(nums[i]);
            if (currSum > halfSum) {
                break;
            }
        }
        
        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n \log n)$
  - Sorting the array of length $n$ takes $\mathcal{O}(n \log n)$.
  - Summation and prefix accumulation take $\mathcal{O}(n)$.
  *(Can be optimized to $\mathcal{O}(n)$ using counting sort since $1 \le \text{nums}[i] \le 100$).*
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - In-place sorting and return vector memory.

---

### Takeaway Pattern & Interview Traps

- **Condition Simplification:** Always simplify relative sum conditions: $\text{sum}_{\text{selected}} > \text{sum}_{\text{remaining}} \iff \text{sum}_{\text{selected}} > \frac{\text{total\_sum}}{2}$.
- **Strict Inequality:** The problem specifies *strictly greater*. A tie ($\text{sum}_{\text{selected}} == \frac{\text{total\_sum}}{2}$) is invalid; you must pick at least one more element.