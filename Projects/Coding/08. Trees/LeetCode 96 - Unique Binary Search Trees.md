---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 96: Unique Binary Search Trees"
tags:
  - leetcode
  - coding
  - trees
  - dynamic-programming
  - math
  - amazon
  - google
---

# LeetCode 96: Unique Binary Search Trees

**Target Companies:** Google, Amazon, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Catalan Numbers / Combinatorial Decomposition

---

### Problem Statement

Given an integer `n`, return the number of structurally unique **BST's** (binary search trees) which has exactly `n` nodes of unique values from `1` to `n`.

---

### Input & Output Formats & Constraints

- **Input:** `n: int`
- **Output:** `int` — Total count of structurally unique BSTs.
- **Constraints:**
  - $1 \le n \le 19$

---

### Key Idea & Intuition

- **Subproblem Decomposition by Root Choice:**
  - Let $G(n)$ be the total number of structurally unique BSTs that can be formed using $n$ distinct keys.
  - If we choose key $i$ ($1 \le i \le n$) as the root of the tree:
    - Keys $\{1, 2, \dots, i - 1\}$ must form the left subtree (size $i - 1$).
    - Keys $\{i + 1, i + 2, \dots, n\}$ must form the right subtree (size $n - i$).
  - Because any valid left subtree can be paired with any valid right subtree independently, the number of unique BSTs with root $i$ is:
    $$F(i, n) = G(i - 1) \times G(n - i)$$
- **Catalan Number Recurrence:**
  - Summing over all possible roots $i \in [1, n]$:
    $$G(n) = \sum_{i=1}^n G(i - 1) \times G(n - i)$$
  - **Base Cases:**
    - $G(0) = 1$: An empty subtree is a single unique valid BST structure (null pointer).
    - $G(1) = 1$: A single node forms exactly 1 unique BST.
  - This sequence generates the Catalan numbers $C_n = \frac{1}{n+1} \binom{2n}{n}$.

---

### Solution Approach (Step-by-Step)

1. Initialize an array `dp` of size $n + 1$ with zeros.
2. Set base cases: `dp[0] = 1`, `dp[1] = 1`.
3. Loop `length` from $2$ to $n$:
   - Loop `root` from $1$ to `length`:
     - `left = root - 1`
     - `right = length - root`
     - `dp[length] += dp[left] * dp[right]`
4. Return `dp[n]`.

---

### Visual Algorithm Walkthrough

```
Computing G(3):
Possible roots: 1, 2, 3

1. Root = 1:
   - Left subtree size: 0  -> dp[0] = 1
   - Right subtree size: 2 -> dp[2] = 2
   - Contribution: 1 * 2 = 2

2. Root = 2:
   - Left subtree size: 1  -> dp[1] = 1
   - Right subtree size: 1 -> dp[1] = 1
   - Contribution: 1 * 1 = 1

3. Root = 3:
   - Left subtree size: 2  -> dp[2] = 2
   - Right subtree size: 0 -> dp[0] = 1
   - Contribution: 2 * 1 = 2

Total dp[3] = 2 + 1 + 2 = 5 unique BSTs.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: $n = 3$
- **Input:** `n = 3`
- **DP State Evaluation Table:**
  | $n$ | Summation Breakdown ($\sum dp[i-1] \times dp[n-i]$) | $dp[n]$ |
  | :--- | :--- | :--- |
  | 0 | Base case (empty tree) | 1 |
  | 1 | Base case (single node) | 1 |
  | 2 | $dp[0] \times dp[1] + dp[1] \times dp[0] = 1 \times 1 + 1 \times 1$ | 2 |
  | 3 | $dp[0] \times dp[2] + dp[1] \times dp[1] + dp[2] \times dp[0] = 1 \cdot 2 + 1 \cdot 1 + 2 \cdot 1$ | **5** |
- **Output:** `5`

#### Example 2: $n = 4$
- **Calculation:**
  $$dp[4] = dp[0]dp[3] + dp[1]dp[2] + dp[2]dp[1] + dp[3]dp[0] = 1(5) + 1(2) + 2(1) + 5(1) = 14$$
- **Output:** `14`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class Solution:
    def numTrees(self, n: int) -> int:
        # dp[i] stores the number of unique BSTs of length i
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1
        
        for length in range(2, n + 1):
            for root in range(1, length + 1):
                left = root - 1
                right = length - root
                dp[length] += dp[left] * dp[right]
                
        return dp[n]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    int numTrees(int n) {
        std::vector<int> dp(n + 1, 0);
        dp[0] = 1;
        dp[1] = 1;

        for (int length = 2; length <= n; ++length) {
            for (int root = 1; root <= length; ++root) {
                int left = root - 1;
                int right = length - root;
                dp[length] += dp[left] * dp[right];
            }
        }

        return dp[n];
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;

        for (int length = 2; length <= n; length++) {
            for (int root = 1; root <= length; root++) {
                int left = root - 1;
                int right = length - root;
                dp[length] += dp[left] * dp[right];
            }
        }

        return dp[n];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n^2)$ — Outer loop runs from $2$ to $n$. Inner loop runs $length$ times. Total iterations $= \sum_{k=2}^n k = \frac{n(n+1)}{2} - 1 = \mathcal{O}(n^2)$. For $n \le 19$, operations $\le 200$, effectively instantaneous.
  *(Note: Using the closed-form Catalan formula $C_n = \frac{1}{n+1} \binom{2n}{n}$ solves the problem in $\mathcal{O}(n)$ time and $\mathcal{O}(1)$ space).*
- **Space Complexity:** $\mathcal{O}(n)$ — 1D dynamic programming table of size $n + 1$.

---

### Takeaway Pattern & Interview Traps

1. **The Empty Tree Base Case:** $dp[0] = 1$ is mandatory. If $dp[0]$ were initialized to $0$, any tree whose left or right subtree was empty would multiply by zero and wipe out the valid configurations of the opposing subtree.
2. **Values Don't Matter, Count Does:** Whether keys are $\{1, 2, 3\}$ or $\{100, 200, 300\}$, any set of $k$ sorted distinct numbers produces the exact same number of unique BST structures: $G(k)$.
3. **Difference from LeetCode 95:** LeetCode 96 asks only for the count ($G(n)$), while LeetCode 95 asks for all actual tree instances.