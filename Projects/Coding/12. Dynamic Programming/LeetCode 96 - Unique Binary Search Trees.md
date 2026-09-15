---
date: "2026-09-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 96: Unique Binary Search Trees"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - math
  - tree
  - binary-search-tree
  - combinatorics
  - google
  - amazon
  - meta
---

# LeetCode 96: Unique Binary Search Trees

**Target Companies:** Google, Amazon, Meta, Microsoft, Apple, Adobe  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Catalan Numbers / Combinatorics

---

### Problem Statement

Given an integer `n`, return *the number of structurally unique **BST's** (binary search trees) which has exactly `n` nodes of unique values from `1` to `n`*.

---

### Input & Output Formats & Constraints

- **Input:** `n: int`
- **Output:** `int`
- **Constraints:**
  - `1 <= n <= 19`
  - The answer fits within a standard 32-bit signed integer.

---

### Key Idea & Intuition

Any Binary Search Tree with $n$ nodes must choose one of the values $i \in \{1, 2, \dots, n\}$ to serve as the root.

Due to the strict BST ordering invariant:
1. All nodes with values less than $i$ (values $1, 2, \dots, i - 1$) must reside in the **left subtree**. The number of such nodes is $i - 1$.
2. All nodes with values greater than $i$ (values $i + 1, \dots, n$) must reside in the **right subtree**. The number of such nodes is $n - i$.

Let $G(n)$ denote the total number of unique BSTs that can be formed using $n$ distinct keys.  
Notice that the structural shapes of a BST depend strictly on the **count** of keys, not their exact numerical values (e.g., $\{1, 2, 3\}$ and $\{4, 5, 6\}$ generate the exact same structural combinations).

Therefore, for a chosen root $i$:
$$\text{Trees with root } i = G(i - 1) \times G(n - i)$$

Summing across all possible root candidates $i \in [1, n]$:
$$G(n) = \sum_{i=1}^n G(i - 1) \times G(n - i)$$

#### Base Cases:
- $G(0) = 1$: An empty tree is uniquely defined by a single null pointer.
- $G(1) = 1$: A single node has exactly 1 valid tree.

This recurrence defines the **Catalan Numbers**:
$$C_n = \frac{1}{n+1}\binom{2n}{n} = \prod_{k=1}^n \frac{n + k}{k}$$

We can solve this problem in two ways:
1. **Dynamic Programming:** Build up $G(k)$ from $k = 2$ to $n$ in $\mathcal{O}(n^2)$ time and $\mathcal{O}(n)$ space.
2. **Direct Catalan Formula:** Iteratively compute $C_n$ in $\mathcal{O}(n)$ time and $\mathcal{O}(1)$ space.

---

### Solution Approach (Step-by-Step)

#### Method 1: Dynamic Programming ($\mathcal{O}(n^2)$)
1. Allocate an array `dp` of size $n + 1$, initialized with zeros.
2. Set base cases: `dp[0] = 1` and `dp[1] = 1`.
3. Loop for tree size `len` from $2$ to $n$:
   - Loop for root position `root` from $1$ to `len`:
     - Left subtree size = `root - 1`
     - Right subtree size = `len - root`
     - `dp[len] += dp[root - 1] * dp[len - root]`
4. Return `dp[n]`.

#### Method 2: Mathematical Direct Computation ($\mathcal{O}(n)$)
Using the iterative Catalan recurrence:
$$C_0 = 1, \quad C_k = C_{k-1} \times \frac{2(2k - 1)}{k + 1}$$
We compute the value iteratively using 64-bit integers to prevent intermediate overflow.

---

### Visual Algorithm Walkthrough

#### Constructing $G(3)$ from Subproblems

```
Base: G(0) = 1, G(1) = 1

For len = 2:
  root = 1: G(0) * G(1) = 1 * 1 = 1
  root = 2: G(1) * G(0) = 1 * 1 = 1
  dp[2] = 1 + 1 = 2

For len = 3:
  root = 1 (left size 0, right size 2): dp[0] * dp[2] = 1 * 2 = 2
      1              1
       \              \
        2              3
         \            /
          3          2

  root = 2 (left size 1, right size 1): dp[1] * dp[1] = 1 * 1 = 1
        2
       / \
      1   3

  root = 3 (left size 2, right size 0): dp[2] * dp[0] = 2 * 1 = 2
        3            3
       /            /
      1            2
       \          /
        2        1

  dp[3] = 2 + 1 + 2 = 5
```

---

### Solved Examples with Multiple Inputs

| $n$ | Recurrence Expansion | Output ($C_n$) |
| :--- | :--- | :--- |
| `1` | $G(1) = 1$ | `1` |
| `2` | $G(0)G(1) + G(1)G(0) = 1 + 1$ | `2` |
| `3` | $G(0)G(2) + G(1)G(1) + G(2)G(0) = 2 + 1 + 2$ | `5` |
| `4` | $G(0)G(3) + G(1)G(2) + G(2)G(1) + G(3)G(0) = 5 + 2 + 2 + 5$ | `14` |
| `5` | $14 + 5 + 4 + 5 + 14$ | `42` |
| `10` | $C_{10}$ | `16,796` |
| `19` | $C_{19}$ (Maximum constraint) | `1,767,263,190` |

*Note: For $n = 19$, $C_{19} = 1,767,263,190 < 2^{31} - 1$, fitting within signed 32-bit integer limits.*

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def numTrees(self, n: int) -> int:
        # dp[i] represents the number of unique BSTs that can be formed with i nodes
        dp = [0] * (n + 1)
        dp[0] = 1  # Base case: empty tree
        dp[1] = 1  # Base case: single node tree
        
        for length in range(2, n + 1):
            for root in range(1, length + 1):
                left = root - 1
                right = length - root
                dp[length] += dp[left] * dp[right]
                
        return dp[n]

    def numTreesMath(self, n: int) -> int:
        """Alternative O(n) math solution using Catalan formula."""
        catalan = 1
        for k in range(1, n + 1):
            catalan = catalan * (4 * k - 2) // (k + 1)
        return catalan
```

#### C++17
```cpp
#include <vector>

class Solution {
public:
    int numTrees(int n) {
        // dp[i] stores the number of unique BSTs with i nodes
        std::vector<int> dp(n + 1, 0);
        dp[0] = 1;
        dp[1] = 1;

        for (int len = 2; len <= n; ++len) {
            for (int root = 1; root <= len; ++root) {
                int left = root - 1;
                int right = len - root;
                dp[len] += dp[left] * dp[right];
            }
        }

        return dp[n];
    }

    // Alternative O(n) math solution
    int numTreesMath(int n) {
        long long c = 1;
        for (int k = 1; k <= n; ++k) {
            c = c * 2 * (2 * k - 1) / (k + 1);
        }
        return static_cast<int>(c);
    }
};
```

#### Java 17
```java
class Solution {
    public int numTrees(int n) {
        // dp[i] stores the number of unique BSTs with i nodes
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;

        for (int len = 2; len <= n; len++) {
            for (int root = 1; root <= len; root++) {
                int left = root - 1;
                int right = len - root;
                dp[len] += dp[left] * dp[right];
            }
        }

        return dp[n];
    }

    // Alternative O(n) math solution
    public int numTreesMath(int n) {
        long c = 1;
        for (int k = 1; k <= n; k++) {
            c = c * 2 * (2 * k - 1) / (k + 1);
        }
        return (int) c;
    }
}
```

---

### Complexity Analysis

- **Dynamic Programming Approach:**
  - **Time Complexity:** $\mathcal{O}(n^2)$. We use two nested loops: the outer loop runs from $2$ to $n$, and the inner loop runs from $1$ to $len$, yielding $\sum_{len=2}^n len = \frac{n(n+1)}{2} - 1 = \mathcal{O}(n^2)$ iterations. For $n \le 19$, this requires under 200 operations.
  - **Space Complexity:** $\mathcal{O}(n)$ auxiliary space to store the DP table of size $n+1$.

- **Mathematical Approach:**
  - **Time Complexity:** $\mathcal{O}(n)$ single loop calculating the product for $k \in [1, n]$.
  - **Space Complexity:** $\mathcal{O}(1)$ scalar storage.

---

### Takeaway Pattern & Interview Traps

1. **Symmetry Optimization:**
   - Notice that $G(i - 1) \times G(n - i) = G(n - i) \times G(i - 1)$. In DP calculations, one could compute up to $\lceil n / 2 \rceil$ and double the symmetric terms to save half the multiplications.
2. **Intermediate Integer Overflow in Catalan Formula:**
   - When computing $C_k = C_{k-1} \times \frac{2(2k - 1)}{k + 1}$ directly, multiplying before dividing requires 64-bit integer types (`long long` in C++, `long` in Java) because the numerator $C_{k-1} \times (4k - 2)$ can temporarily exceed $2^{31} - 1$ even though the final result for $n \le 19$ fits in a 32-bit signed integer.
3. **Problem Type Recognition:**
   - Any problem partitioning $n$ elements into two independent ordered structures across a split point $k$ (e.g., full binary trees, valid parenthesis combinations, non-crossing chords in a circle, polygon triangulations) follows the Catalan recurrence.