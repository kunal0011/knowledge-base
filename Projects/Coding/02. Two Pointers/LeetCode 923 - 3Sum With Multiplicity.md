---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 923: 3Sum With Multiplicity"
tags:
  - leetcode
  - coding
  - two-pointers
  - combinatorics
  - counting
  - google
  - amazon
---

# LeetCode 923: 3Sum With Multiplicity

**Target Companies:** Google, Amazon, Bloomberg  
**Difficulty:** Medium  
**Topic:** Two Pointers / Combinatorics with Multiplicity Deduplication  

---

### Problem Statement

Given an integer array `arr`, and an integer `target`, return the number of tuples `(i, j, k)` such that:
- $0 \le i < j < k < \text{arr.length}$
- `arr[i] + arr[j] + arr[k] == target`

As the answer can be very large, return it **modulo $10^9 + 7$**.

---

### Input & Output Formats & Constraints

- **Input:** `arr: List[int]`, `target: int`
- **Output:** `int` (total valid index triplets modulo $10^9 + 7$)
- **Constraints:**
  - $3 \le \text{arr.length} \le 3000$
  - $0 \le \text{arr}[i] \le 100$
  - $0 \le target \le 300$

---

### Key Idea & Intuition

Unlike standard 3Sum where duplicates are simply discarded, this problem requires counting every valid **combination of indices**.

By sorting `arr`:
1. We iterate over index $i \in [0, n - 3]$ and set up two pointers: `left = i + 1` and `right = n - 1`.
2. For each step, let `current_sum = arr[i] + arr[left] + arr[right]`:
   - If `current_sum < target`: advance `left++`.
   - If `current_sum > target`: decrement `right--`.
   - If `current_sum == target`: we have reached a match! Handling duplicates requires combinatorics:
     - **Case 1: $arr[left] \ne arr[right]$**
       - Count how many times $arr[left]$ appears contiguously ($c_1$).
       - Count how many times $arr[right]$ appears contiguously ($c_2$).
       - The total combinations contributed by these values is $c_1 \times c_2$.
       - Advance `left += c1` and `right -= c2`.
     - **Case 2: $arr[left] == arr[right]$**
       - Every element from index `left` to index `right` inclusive has the exact same value.
       - Let $k = right - left + 1$.
       - Any pair chosen from these $k$ elements sums to $target - arr[i]$.
       - Number of combinations is $\binom{k}{2} = \frac{k(k - 1)}{2}$.
       - Break the inner loop, since no other pairs exist for this $i$.

---

### Solution Approach (Step-by-Step)

1. Sort `arr` in ascending order.
2. Initialize `ans = 0`, `MOD = 1_000_000_007`.
3. Loop $i$ from $0$ to $n - 3$:
   - `left = i + 1`, `right = n - 1`.
   - While `left < right`:
     - `s = arr[i] + arr[left] + arr[right]`.
     - If `s < target`: `left += 1`.
     - Else if `s > target`: `right -= 1`.
     - Else:
       - If `arr[left] != arr[right]`:
         - Count duplicates of `arr[left]` $\to c_1$.
         - Count duplicates of `arr[right]` $\to c_2$.
         - `ans = (ans + c1 * c2) % MOD`.
         - `left += c1`, `right -= c2`.
       - Else:
         - `k = right - left + 1`.
         - `ans = (ans + k * (k - 1) // 2) % MOD`.
         - Break inner while-loop.
4. Return `ans`.

---

### Visual Algorithm Walkthrough

```
arr = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5], target = 8
Sorted array has 10 elements.

Iteration i = 0 (arr[0] = 1):
  Remaining needed: target - arr[0] = 7.
  left = 1 (arr[1] = 1), right = 9 (arr[9] = 5)
  arr[left] + arr[right] = 1 + 5 = 6 < 7 -> left++
  left = 2 (arr[2] = 2), right = 9 (arr[9] = 5)
  arr[left] + arr[right] = 2 + 5 = 7 == 7 (MATCH!)

  Case 1: arr[left] (2) != arr[right] (5):
    - arr[2] and arr[3] are both 2 -> c1 = 2
    - arr[8] and arr[9] are both 5 -> c2 = 2
    - Combinations added: 2 * 2 = 4. (Tuples: (0,2,8), (0,2,9), (0,3,8), (0,3,9))
    - Advance left past 2s (left = 4), right past 5s (right = 7).

  Now left = 4 (arr[4] = 3), right = 7 (arr[7] = 4):
  arr[4] + arr[7] = 3 + 4 = 7 == 7 (MATCH!)
  Case 1: arr[left] (3) != arr[right] (4):
    - arr[4] and arr[5] are both 3 -> c1 = 2
    - arr[6] and arr[7] are both 4 -> c2 = 2
    - Combinations added: 2 * 2 = 4.
    - Advance left to 6, right to 5 (left > right -> loop finishes for i = 0).

Continue for all i:
Total combinations summed = 20.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Multiplicity
- **Input:** `arr = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]`, `target = 8`
- **Output:** `20`

#### Example 2: All Elements Identical ($arr[left] == arr[right]$)
- **Input:** `arr = [1, 1, 2, 2, 2, 2]`, `target = 5`
- **Trace:**
  - When $arr[i] = 1$, we need two elements summing to 4.
  - The remaining elements are all `2`.
  - There are four 2s: $\binom{4}{2} = \frac{4 \times 3}{2} = 6$ ways.
  - With two 1s, total ways = $2 \times 6 = 12$.
- **Output:** `12`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def threeSumMulti(self, arr: List[int], target: int) -> int:
        MOD = 1_000_000_007
        arr.sort()
        n = len(arr)
        ans = 0
        
        for i in range(n - 2):
            left = i + 1
            right = n - 1
            rem = target - arr[i]
            
            while left < right:
                total = arr[left] + arr[right]
                if total < rem:
                    left += 1
                elif total > rem:
                    right -= 1
                else:
                    if arr[left] != arr[right]:
                        c1 = 1
                        while left + 1 < right and arr[left] == arr[left + 1]:
                            c1 += 1
                            left += 1
                        c2 = 1
                        while right - 1 > left and arr[right] == arr[right - 1]:
                            c2 += 1
                            right -= 1
                        ans = (ans + c1 * c2) % MOD
                        left += 1
                        right -= 1
                    else:
                        # All elements between left and right are equal
                        k = right - left + 1
                        ans = (ans + (k * (k - 1) // 2)) % MOD
                        break
                        
        return ans
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int threeSumMulti(std::vector<int>& arr, int target) {
        const int MOD = 1e9 + 7;
        std::sort(arr.begin(), arr.end());
        int n = arr.size();
        long long ans = 0;

        for (int i = 0; i < n - 2; ++i) {
            int left = i + 1;
            int right = n - 1;
            int rem = target - arr[i];

            while (left < right) {
                int total = arr[left] + arr[right];
                if (total < rem) {
                    left++;
                } else if (total > rem) {
                    right--;
                } else {
                    if (arr[left] != arr[right]) {
                        int c1 = 1;
                        while (left + 1 < right && arr[left] == arr[left + 1]) {
                            c1++;
                            left++;
                        }
                        int c2 = 1;
                        while (right - 1 > left && arr[right] == arr[right - 1]) {
                            c2++;
                            right--;
                        }
                        ans = (ans + 1LL * c1 * c2) % MOD;
                        left++;
                        right--;
                    } else {
                        long long k = right - left + 1;
                        ans = (ans + (k * (k - 1) / 2)) % MOD;
                        break;
                    }
                }
            }
        }

        return static_cast<int>(ans);
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public int threeSumMulti(int[] arr, int target) {
        final int MOD = 1_000_000_007;
        Arrays.sort(arr);
        int n = arr.length;
        long ans = 0;

        for (int i = 0; i < n - 2; i++) {
            int left = i + 1;
            int right = n - 1;
            int rem = target - arr[i];

            while (left < right) {
                int total = arr[left] + arr[right];
                if (total < rem) {
                    left++;
                } else if (total > rem) {
                    right--;
                } else {
                    if (arr[left] != arr[right]) {
                        int c1 = 1;
                        while (left + 1 < right && arr[left] == arr[left + 1]) {
                            c1++;
                            left++;
                        }
                        int c2 = 1;
                        while (right - 1 > left && arr[right] == arr[right - 1]) {
                            c2++;
                            right--;
                        }
                        ans = (ans + (long) c1 * c2) % MOD;
                        left++;
                        right--;
                    } else {
                        long k = right - left + 1;
                        ans = (ans + (k * (k - 1) / 2)) % MOD;
                        break;
                    }
                }
            }
        }

        return (int) ans;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N^2)$ — Outer loop runs $O(N)$ iterations, and the inner two-pointer search sweeps across the rest of the array in $O(N)$ time.
- **Space Complexity:** $O(1)$ auxiliary space beyond the in-place sort.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** 3Sum with Combinatorial Duplicates Handling ($\binom{k}{2}$ when $arr[left] == arr[right]$, and $c_1 \times c_2$ when distinct).
- **Trap:** Forgetting 64-bit integer cast before multiplying counts in C++ and Java (`1LL * c1 * c2`), which can cause intermediate 32-bit overflow before modulo.