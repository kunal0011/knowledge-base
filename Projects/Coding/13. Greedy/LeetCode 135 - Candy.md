---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 135: Candy"
tags:
  - leetcode
  - coding
  - greedy
  - arrays
  - amazon
  - google
  - meta
---

# LeetCode 135: Candy

**Target Companies:** Google (Signature Greedy Hard), Amazon, Meta, Apple, Microsoft  
**Difficulty:** Hard  
**Topic:** Two-Pass Greedy Slope Invariant / Peak & Valley Analysis

---

### Problem Statement

There are $n$ children standing in a line. Each child is assigned a rating value given in the integer array `ratings`.

You are giving candies to these children subjected to the following requirements:
1. Each child must have at least one candy.
2. Children with a higher rating get more candies than their neighbors.

Return *the minimum number of candies you need to have to distribute the candies to the children*.

---

### Input & Output Formats & Constraints

- **Input:** `ratings: List[int]`
- **Output:** `int` (minimum total candies)
- **Constraints:**
  - $n == \text{ratings.length}$
  - $1 \le n \le 2 \times 10^4$
  - $0 \le \text{ratings}[i] \le 2 \times 10^4$

---

### Key Idea & Intuition

#### 1. Decoupling Bilateral Constraints:
Each child $i$ must satisfy two independent directional conditions:
1. **Left-Neighbor Condition:** If $\text{ratings}[i] > \text{ratings}[i - 1]$, then $\text{candies}[i] > \text{candies}[i - 1]$.
2. **Right-Neighbor Condition:** If $\text{ratings}[i] > \text{ratings}[i + 1]$, then $\text{candies}[i] > \text{candies}[i + 1]$.

Attempting to resolve both constraints simultaneously in a single pass leads to complex lookahead and lookback state tracking. However, because both constraints are independent local monotonicity requirements, we can decompose the problem into **two separate greedy passes**:

#### 2. Pass 1: Left-to-Right
- Initialize every child with $1$ candy (satisfying Requirement 1).
- Traverse from index $1$ to $n - 1$:
  - If $\text{ratings}[i] > \text{ratings}[i - 1]$, give child $i$ one more candy than their left neighbor:
    $$\text{candies}[i] = \text{candies}[i - 1] + 1$$

#### 3. Pass 2: Right-to-Left
- Traverse from index $n - 2$ down to $0$:
  - If $\text{ratings}[i] > \text{ratings}[i + 1]$, child $i$ must have more candies than child $i + 1$.
  - To preserve the left-neighbor condition already satisfied in Pass 1, we set:
    $$\text{candies}[i] = \max(\text{candies}[i], \text{candies}[i + 1] + 1)$$
- Taking the maximum guarantees that **both** neighbor constraints are simultaneously satisfied with the minimum possible candy count!

---

### Solution Approach (Step-by-Step)

1. **Initialize Candies Array:**
   - Allocate `candies` of size $n$, filled with 1s.
2. **Left-to-Right Scan:**
   - For $i$ from 1 to $n - 1$:
     - If `ratings[i] > ratings[i - 1]`:
       - `candies[i] = candies[i - 1] + 1`
3. **Right-to-Left Scan:**
   - For $i$ from $n - 2$ down to 0:
     - If `ratings[i] > ratings[i + 1]`:
       - `candies[i] = max(candies[i], candies[i + 1] + 1)`
4. **Sum and Return:**
   - Sum all elements in `candies` and return the total.

---

### Visual Algorithm Walkthrough

#### Example: `ratings = [1, 0, 2]`

```
Initial: [1, 1, 1]

Pass 1 (Left to Right):
  i=1: ratings[1] (0) not > ratings[0] (1) -> candies[1] remains 1
  i=2: ratings[2] (2) > ratings[1] (0)     -> candies[2] = candies[1] + 1 = 2
  Array after Pass 1: [1, 1, 2]

Pass 2 (Right to Left):
  i=1: ratings[1] (0) not > ratings[2] (2) -> candies[1] remains 1
  i=0: ratings[0] (1) > ratings[1] (0)     -> candies[0] = max(1, candies[1] + 1) = max(1, 2) = 2
  Array after Pass 2: [2, 1, 2]

Total candies = 2 + 1 + 2 = 5.
```

---

### Solved Examples with Multiple Inputs

| Input `ratings` | After Pass 1 | After Pass 2 | Total Candies | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| `[1, 0, 2]` | `[1, 1, 2]` | `[2, 1, 2]` | `5` | Peak at 0 and 2, valley at 1 |
| `[1, 2, 2]` | `[1, 2, 1]` | `[1, 2, 1]` | `4` | Equal rating does not require more candies |
| `[1, 2, 87, 87, 87, 2, 1]` | `[1, 2, 3, 1, 1, 1, 1]` | `[1, 2, 3, 1, 3, 2, 1]` | `13` | Plateau with steep descent |
| `[5, 4, 3, 2, 1]` | `[1, 1, 1, 1, 1]` | `[5, 4, 3, 2, 1]` | `15` | Pure decreasing sequence |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        candies = [1] * n
        
        # Pass 1: Left to right
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1
                
        # Pass 2: Right to left
        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                candies[i] = max(candies[i], candies[i + 1] + 1)
                
        return sum(candies)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <numeric>
#include <algorithm>

class Solution {
public:
    int candy(std::vector<int>& ratings) {
        int n = ratings.size();
        std::vector<int> candies(n, 1);

        // Left to right
        for (int i = 1; i < n; ++i) {
            if (ratings[i] > ratings[i - 1]) {
                candies[i] = candies[i - 1] + 1;
            }
        }

        // Right to left
        for (int i = n - 2; i >= 0; --i) {
            if (ratings[i] > ratings[i + 1]) {
                candies[i] = std::max(candies[i], candies[i + 1] + 1);
            }
        }

        return std::accumulate(candies.begin(), candies.end(), 0);
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public int candy(int[] ratings) {
        int n = ratings.length;
        int[] candies = new int[n];
        Arrays.fill(candies, 1);

        // Left to right
        for (int i = 1; i < n; i++) {
            if (ratings[i] > ratings[i - 1]) {
                candies[i] = candies[i - 1] + 1;
            }
        }

        // Right to left
        for (int i = n - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1]) {
                candies[i] = Math.max(candies[i], candies[i + 1] + 1);
            }
        }

        int total = 0;
        for (int c : candies) {
            total += c;
        }
        return total;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$, where $n$ is the length of `ratings`. We make two linear sweeps over the array, each taking $\mathcal{O}(n)$ time.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space to store the candy count allocated to each child.

---

### Takeaway Pattern & Interview Traps

1. **Equal Ratings Trap:**
   - If two adjacent children have the **same** rating (`ratings[i] == ratings[i-1]`), the rule does NOT require them to have the same number of candies. One child can have 1 candy while their neighbor has 5 candies. Only strictly higher ratings demand more candies.
2. **`max()` in Second Pass is Mandatory:**
   - In Pass 2, using `candies[i] = candies[i + 1] + 1` instead of `candies[i] = max(candies[i], candies[i + 1] + 1)` would destroy valid higher candy allocations achieved during the first left-to-right pass.
3. **$\mathcal{O}(1)$ Space Follow-Up (One-Pass Slope Tracking):**
   - The problem can also be solved in $\mathcal{O}(1)$ space by tracking up-slopes, down-slopes, and peak heights directly (counting triangles of candies).
