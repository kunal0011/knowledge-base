---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 312: Burst Balloons"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - array
  - amazon
  - google
  - meta
  - microsoft
---

# LeetCode 312: Burst Balloons

**Target Companies:** Amazon, Google, Meta, Microsoft, Bloomberg, Apple  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / Interval DP  

---

### Problem Statement

You are given `n` balloons, indexed from `0` to `n - 1`. Each balloon is painted with a number on it represented by an array `nums`. You are asked to burst all the balloons.

If you burst the $i^{\text{th}}$ balloon, you will get `nums[i - 1] * nums[i] * nums[i + 1]` coins. If `i - 1` or `i + 1` goes out of bounds of the array, then treat it as if there is a balloon with a `1` painted on it.

Return *the maximum coins you can collect by bursting the balloons wisely*.

---

### Input & Output Formats & Constraints

- **Input:** An integer array `nums` of length $n$ ($1 \le n \le 300$).
- **Output:** An integer representing the maximum coins collectible.
- **Constraints:**
  - `n == nums.length`
  - `1 <= n <= 300`
  - `0 <= nums[i] <= 100`

---

### Key Idea & Intuition

#### Why "First Balloon Burst" Fails
If we attempt to choose which balloon to burst first, bursting balloon $k$ changes the adjacency structure for all remaining balloons: $nums[k-1]$ and $nums[k+1]$ become immediate neighbors. This dynamic dependency ties the left and right subproblems together, destroying optimal substructure.

#### The "Last Balloon Burst" Inversion Principle
Instead of asking which balloon bursts first, we invert our perspective:
> **Which balloon $k$ is burst LAST in the open interval $(i, j)$?**

Why this inversion works:
1. If balloon $k$ ($i < k < j$) is the **very last** balloon burst among all balloons strictly between $i$ and $j$, then every other balloon in $(i, k)$ and $(k, j)$ has already been popped.
2. Therefore, when balloon $k$ is popped, its remaining adjacent neighbors are **guaranteed to be the outer boundary balloons $i$ and $j$**!
3. The coins gained from bursting balloon $k$ last is strictly:
   $$\text{coins} = arr[i] \times arr[k] \times arr[j]$$
4. This completely decouples the remaining subproblems into two independent sub-intervals:
   - Burst all balloons in open interval $(i, k)$
   - Burst all balloons in open interval $(k, j)$

#### Recurrence Relation
Pad `nums` with boundary `1`s: `arr = [1] + nums + [1]`, with new length $N = n + 2$.
Let $dp[i][j]$ be the maximum coins obtained by bursting all balloons strictly between index $i$ and index $j$:
$$dp[i][j] = \max_{i < k < j} \Big(dp[i][k] + dp[k][j] + arr[i] \times arr[k] \times arr[j]\Big)$$
Base Case: For adjacent indices ($j = i + 1$), there are no balloons between them, so $dp[i][j] = 0$.

---

### Solution Approach (Step-by-Step)

1. **Virtual Boundary Padding:**
   - Create `arr = [1] + nums + [1]`.
   - Length $N = n + 2$.
2. **Initialize DP Table:**
   - Create 2D array `dp` of size $N \times N$ filled with 0.
3. **Iterate by Interval Length:**
   - For `length` from 2 to $N - 1$:
     - For $i$ from 0 to $N - length - 1$:
       - $j = i + length$
       - For $k$ from $i + 1$ to $j - 1$:
         - $dp[i][j] = \max(dp[i][j], \, dp[i][k] + dp[k][j] + arr[i] \times arr[k] \times arr[j])$
4. **Return Result:**
   - Return $dp[0][N - 1]$.

---

### Visual Algorithm Walkthrough

#### Trace for `nums = [3, 1, 5, 8]`
```
Padded Array: arr = [1, 3, 1, 5, 8, 1], N = 6
Indices:             0  1  2  3  4  5

Base Cases (length = 1): dp[i][i+1] = 0

Length = 2 (Single internal balloon):
- (0, 2), k=1 ('3'): 0 + 0 + arr[0]*arr[1]*arr[2] = 1*3*1 = 3
- (1, 3), k=2 ('1'): 0 + 0 + arr[1]*arr[2]*arr[3] = 3*1*5 = 15
- (2, 4), k=3 ('5'): 0 + 0 + arr[2]*arr[3]*arr[4] = 1*5*8 = 40
- (3, 5), k=4 ('8'): 0 + 0 + arr[3]*arr[4]*arr[5] = 5*8*1 = 40

Length = 3 (Two internal balloons):
- Interval (1, 4), balloons {2, 3} (values [1, 5]):
    If k=2 (burst '1' last): dp[1][2] + dp[2][4] + arr[1]*arr[2]*arr[4]
                            = 0 + 40 + (3*1*8) = 64
    If k=3 (burst '5' last): dp[1][3] + dp[3][4] + arr[1]*arr[3]*arr[4]
                            = 15 + 0 + (3*5*8) = 135
    dp[1][4] = max(64, 135) = 135

After computing all lengths up to length = 5:
Full Interval (0, 5):
  k = 1 is popped last -> coins = 167.

Final Answer: dp[0][5] = 167.
Burst sequence: Pop 1 ('1') -> Pop 5 ('5') -> Pop 3 ('3') -> Pop 8 ('8')
```

---

### Solved Examples with Multiple Inputs

| `nums` | Padded Array `arr` | Optimal Burst Sequence | Output |
|---|---|---|---|
| `[3, 1, 5, 8]` | `[1, 3, 1, 5, 8, 1]` | $1 \to 5 \to 3 \to 8$ | `167` |
| `[1, 5]` | `[1, 1, 5, 1]` | $1 \to 5$ | `10` ($1 \times 1 \times 5 + 1 \times 5 \times 1 = 10$) |
| `[9]` | `[1, 9, 1]` | Single balloon: $1 \times 9 \times 1$ | `9` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def maxCoins(self, nums: list[int]) -> int:
        # Pad with 1 on both ends
        arr: list[int] = [1] + nums + [1]
        n: int = len(arr)
        
        # dp[i][j] stores max coins from balloons strictly between i and j
        dp: list[list[int]] = [[0] * n for _ in range(n)]
        
        # length is the distance between boundary indices i and j
        for length in range(2, n):
            for i in range(n - length):
                j = i + length
                for k in range(i + 1, j):
                    coins = dp[i][k] + dp[k][j] + arr[i] * arr[k] * arr[j]
                    if coins > dp[i][j]:
                        dp[i][j] = coins
                        
        return dp[0][n - 1]
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxCoins(const std::vector<int>& nums) {
        int m = static_cast<int>(nums.size());
        std::vector<int> arr(m + 2, 1);
        for (int i = 0; i < m; ++i) {
            arr[i + 1] = nums[i];
        }

        int n = m + 2;
        std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));

        // Iterate across increasing interval spans
        for (int length = 2; length < n; ++length) {
            for (int i = 0; i < n - length; ++i) {
                int j = i + length;
                for (int k = i + 1; k < j; ++k) {
                    dp[i][j] = std::max(dp[i][j],
                        dp[i][k] + dp[k][j] + arr[i] * arr[k] * arr[j]);
                }
            }
        }

        return dp[0][n - 1];
    }
};
```

#### Java 17
```java
class Solution {
    public int maxCoins(int[] nums) {
        int m = nums.length;
        int[] arr = new int[m + 2];
        arr[0] = 1;
        arr[m + 1] = 1;
        System.arraycopy(nums, 0, arr, 1, m);

        int n = m + 2;
        int[][] dp = new int[n][n];

        // Interval DP by increasing length
        for (int length = 2; length < n; length++) {
            for (int i = 0; i < n - length; i++) {
                int j = i + length;
                for (int k = i + 1; k < j; k++) {
                    dp[i][j] = Math.max(dp[i][j],
                        dp[i][k] + dp[k][j] + arr[i] * arr[k] * arr[j]);
                }
            }
        }

        return dp[0][n - 1];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n^3)$, where $n$ is the number of balloons. The three nested loops iterate over interval lengths ($n$), starting points ($n$), and partition split points ($n$). With $n \le 300$, the total number of operations is approximately $\frac{300^3}{6} \approx 4.5 \times 10^6$, running in under $40$ ms.
- **Space Complexity:** $\mathcal{O}(n^2)$ auxiliary space for the $(n + 2) \times (n + 2)$ DP table.

---

### Takeaway Pattern & Interview Traps

1. **Why "Last" is the Key to Interval DP:** Whenever an operation removes an element and alters adjacency (like bursting balloons or matrix chain multiplication parenthesization), fixing the *last* element guarantees that its neighbors are the static endpoints of the interval.
2. **Open Interval Definition:** Notice that $dp[i][j]$ is defined over the *open* interval $(i, j)$ excluding both $i$ and $j$. This ensures that indices $0$ and $N - 1$ (the virtual boundaries) are never burst.
3. **Loop Ordering:** Interval DP always requires computing smaller sub-interval lengths before larger ones. Looping by `length` guarantees all subproblems $dp[i][k]$ and $dp[k][j]$ are resolved before $dp[i][j]$.