---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 119: Pascal's Triangle II"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - array
  - math
  - amazon
  - google
  - apple
  - microsoft
---

# LeetCode 119: Pascal's Triangle II

**Target Companies:** Amazon, Apple, Google, Microsoft, Bloomberg  
**Difficulty:** Easy  
**Topic:** Dynamic Programming / Array / Math  

---

### Problem Statement

Given an integer `rowIndex`, return the `rowIndex`$^{\text{th}}$ (**0-indexed**) row of **Pascal's triangle**.

In **Pascal's triangle**, each number is the sum of the two numbers directly above it:
```
rowIndex = 0:      [1]
rowIndex = 1:     [1, 1]
rowIndex = 2:    [1, 2, 1]
rowIndex = 3:   [1, 3, 3, 1]
rowIndex = 4:  [1, 4, 6, 4, 1]
```

---

### Input & Output Formats & Constraints

- **Input:** An integer `rowIndex` ($0 \le rowIndex \le 33$).
- **Output:** A list of integers `List[int]` representing the elements of the specified row.
- **Constraints:**
  - `0 <= rowIndex <= 33`
  - **Follow up:** Could you optimize your algorithm to use only $\mathcal{O}(rowIndex)$ extra space?

---

### Key Idea & Intuition

#### The Space Optimization Challenge
In LeetCode 118, we generated the entire triangle using $\mathcal{O}(rowIndex^2)$ space. Here, we are tasked with returning only the single row at index `rowIndex`.

#### In-Place 1D Dynamic Programming
Notice the recurrence relation for any row $i$:
$$\text{row}[j] = \text{row}[j] + \text{row}[j - 1]$$
If we update a single 1D array of size $rowIndex + 1$ from left to right, computing $\text{row}[j]$ would overwrite its value before $\text{row}[j + 1]$ has a chance to use it.

#### Right-to-Left Traversal Invariant
To prevent this corruption, we update the array **from right to left**:
1. Start with $\text{row} = [1] + [0] \times rowIndex$.
2. For each step $i$ from 1 to $rowIndex$:
   - For $j$ from $i$ down to 1:
     $$\text{row}[j] = \text{row}[j] + \text{row}[j - 1]$$
3. Because $j$ decreases, when evaluating $\text{row}[j]$, $\text{row}[j - 1]$ is still in its state from the previous row $i - 1$.
4. This yields a clean in-place dynamic programming solution strictly using $\mathcal{O}(rowIndex)$ auxiliary space.

---

### Solution Approach (Step-by-Step)

1. **Initialize DP Array:**
   - Create a list/vector `row` of size $rowIndex + 1$ initialized with zeros.
   - Set `row[0] = 1`.
2. **Layered In-Place Updates:**
   - For $i$ from 1 to $rowIndex$:
     - For $j$ from $i$ down to 1:
       - `row[j] += row[j - 1]`
3. **Return:**
   - Return `row`.

---

### Visual Algorithm Walkthrough

#### Trace for `rowIndex = 3`
```
Initial: row = [1, 0, 0, 0]

Iteration i = 1:
  j = 1: row[1] = row[1] + row[0] = 0 + 1 = 1
  State: [1, 1, 0, 0]

Iteration i = 2:
  j = 2: row[2] = row[2] + row[1] = 0 + 1 = 1
  j = 1: row[1] = row[1] + row[0] = 1 + 1 = 2
  State: [1, 2, 1, 0]

Iteration i = 3:
  j = 3: row[3] = row[3] + row[2] = 0 + 1 = 1
  j = 2: row[2] = row[2] + row[1] = 1 + 2 = 3
  j = 1: row[1] = row[1] + row[0] = 2 + 1 = 3
  State: [1, 3, 3, 1]

Final Result: [1, 3, 3, 1]
```

---

### Solved Examples with Multiple Inputs

| `rowIndex` | Step-by-Step Row Values | Output |
|---|---|---|
| `0` | `[1]` | `[1]` |
| `1` | `[1, 1]` | `[1, 1]` |
| `3` | `[1, 3, 3, 1]` | `[1, 3, 3, 1]` |
| `4` | `[1, 4, 6, 4, 1]` | `[1, 4, 6, 4, 1]` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def getRow(self, rowIndex: int) -> list[int]:
        # Initialize 1D DP array of required row size
        row: list[int] = [0] * (rowIndex + 1)
        row[0] = 1
        
        for i in range(1, rowIndex + 1):
            # Traverse right-to-left to preserve row[j - 1] from previous layer
            for j in range(i, 0, -1):
                row[j] += row[j - 1]
                
        return row
```

#### C++17
```cpp
#include <vector>

class Solution {
public:
    std::vector<int> getRow(int rowIndex) {
        std::vector<int> row(rowIndex + 1, 0);
        row[0] = 1;
        
        for (int i = 1; i <= rowIndex; ++i) {
            // Update in reverse to compute in-place
            for (int j = i; j > 0; --j) {
                row[j] += row[j - 1];
            }
        }
        
        return row;
    }
};
```

#### Java 17
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<Integer> getRow(int rowIndex) {
        int[] row = new int[rowIndex + 1];
        row[0] = 1;
        
        for (int i = 1; i <= rowIndex; i++) {
            for (int j = i; j > 0; j--) {
                row[j] += row[j - 1];
            }
        }
        
        List<Integer> result = new ArrayList<>(rowIndex + 1);
        for (int val : row) {
            result.add(val);
        }
        
        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(rowIndex^2)$. The nested loops execute $1 + 2 + \dots + rowIndex = \frac{rowIndex \times (rowIndex + 1)}{2}$ additions. For $rowIndex \le 33$, this is at most $561$ operations, running in microseconds.
- **Space Complexity:** $\mathcal{O}(rowIndex)$ auxiliary space. Only a single 1D array of size $rowIndex + 1$ is maintained in memory, strictly fulfilling the follow-up requirement.

---

### Takeaway Pattern & Interview Traps

1. **Why Reverse Traversal is Mandatory:** In 0/1 knapsack and 1D DP transitions where state $dp[j]$ depends on $dp[j - 1]$ from the *previous* layer, iterating forward overwrites $dp[j - 1]$ prematurely. Iterating backward ensures $dp[j - 1]$ remains uncorrupted.
2. **Alternative $\mathcal{O}(k)$ Time Math Approach:**
   - Elements can be computed directly using combinations:
     $$\binom{n}{k} = \binom{n}{k-1} \times \frac{n - k + 1}{k}$$
   - This computes each element in $\mathcal{O}(1)$ time without outer loops, achieving $\mathcal{O}(rowIndex)$ runtime. However, when multiplying large intermediate 32-bit numbers, care must be taken with 64-bit integer overflow.