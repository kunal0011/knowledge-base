---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 118: Pascal's Triangle"
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

# LeetCode 118: Pascal's Triangle

**Target Companies:** Amazon, Apple, Google, Bloomberg, Microsoft, Meta  
**Difficulty:** Easy  
**Topic:** Dynamic Programming / Array / Math  

---

### Problem Statement

Given an integer `numRows`, return the first `numRows` of **Pascal's triangle**.

In **Pascal's triangle**, each number is the sum of the two numbers directly above it:
```
        1
       1 1
      1 2 1
     1 3 3 1
    1 4 6 4 1
```

---

### Input & Output Formats & Constraints

- **Input:** An integer `numRows` ($1 \le numRows \le 30$).
- **Output:** A list of lists of integers `List[List[int]]` where the $i^{\text{th}}$ list contains the $i + 1$ integers of the $i^{\text{th}}$ row of Pascal's triangle.
- **Constraints:**
  - `1 <= numRows <= 30`

---

### Key Idea & Intuition

#### Recurrence & Optimal Substructure
Pascal's Triangle is the canonical geometric illustration of dynamic programming:
1. Each element in the triangle represents the binomial coefficient $\binom{i}{j}$, read as "the number of ways to choose $j$ elements from $i$ elements".
2. Pascal's Identity:
   $$\binom{i}{j} = \binom{i-1}{j-1} + \binom{i-1}{j}$$
3. Boundary Conditions:
   - For every row $i \ge 0$, the first element ($j = 0$) and the last element ($j = i$) are always $1$:
     $$dp[i][0] = 1, \quad dp[i][i] = 1$$
4. Inner Elements ($1 \le j < i$):
   $$dp[i][j] = dp[i-1][j-1] + dp[i-1][j]$$

Each row is constructed purely from the adjacent elements of the immediately preceding row.

---

### Solution Approach (Step-by-Step)

1. **Initialize Triangular List:**
   - Create an empty list `triangle`.
2. **Iterate Through Row Indices:**
   - For $i$ from 0 to $numRows - 1$:
     - Initialize a new row of size $i + 1$ filled with $1$s: `row = [1] * (i + 1)`.
     - For $j$ from 1 to $i - 1$:
       - `row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j]`.
     - Append `row` to `triangle`.
3. **Return:**
   - Return `triangle`.

---

### Visual Algorithm Walkthrough

#### Trace for `numRows = 5`
```
Row 0 (i = 0):
  Size = 1 -> [1]

Row 1 (i = 1):
  Size = 2 -> [1, 1]

Row 2 (i = 2):
  Size = 3 -> [1, ?, 1]
  j = 1: row[1] = triangle[1][0] + triangle[1][1] = 1 + 1 = 2
  Row 2 = [1, 2, 1]

Row 3 (i = 3):
  Size = 4 -> [1, ?, ?, 1]
  j = 1: row[1] = triangle[2][0] + triangle[2][1] = 1 + 2 = 3
  j = 2: row[2] = triangle[2][1] + triangle[2][2] = 2 + 1 = 3
  Row 3 = [1, 3, 3, 1]

Row 4 (i = 4):
  Size = 5 -> [1, ?, ?, ?, 1]
  j = 1: row[1] = triangle[3][0] + triangle[3][1] = 1 + 3 = 4
  j = 2: row[2] = triangle[3][1] + triangle[3][2] = 3 + 3 = 6
  j = 3: row[3] = triangle[3][2] + triangle[3][3] = 3 + 1 = 4
  Row 4 = [1, 4, 6, 4, 1]

Result:
[
  [1],
  [1, 1],
  [1, 2, 1],
  [1, 3, 3, 1],
  [1, 4, 6, 4, 1]
]
```

---

### Solved Examples with Multiple Inputs

| `numRows` | Generated Triangle Rows | Number of Total Elements |
|---|---|---|
| `1` | `[[1]]` | 1 |
| `2` | `[[1], [1, 1]]` | 3 |
| `3` | `[[1], [1, 1], [1, 2, 1]]` | 6 |
| `5` | `[[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]` | 15 |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def generate(self, numRows: int) -> list[list[int]]:
        triangle: list[list[int]] = []
        
        for i in range(numRows):
            # Pre-populate row with 1s of length (i + 1)
            row: list[int] = [1] * (i + 1)
            
            # Compute inner elements
            for j in range(1, i):
                row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j]
                
            triangle.append(row)
            
        return triangle
```

#### C++17
```cpp
#include <vector>

class Solution {
public:
    std::vector<std::vector<int>> generate(int numRows) {
        std::vector<std::vector<int>> triangle;
        triangle.reserve(numRows);
        
        for (int i = 0; i < numRows; ++i) {
            std::vector<int> row(i + 1, 1);
            
            for (int j = 1; j < i; ++j) {
                row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j];
            }
            
            triangle.push_back(std::move(row));
        }
        
        return triangle;
    }
};
```

#### Java 17
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> triangle = new ArrayList<>(numRows);
        
        for (int i = 0; i < numRows; i++) {
            List<Integer> row = new ArrayList<>(i + 1);
            
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) {
                    row.add(1);
                } else {
                    int val = triangle.get(i - 1).get(j - 1) + triangle.get(i - 1).get(j);
                    row.add(val);
                }
            }
            
            triangle.add(row);
        }
        
        return triangle;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(numRows^2)$. The total number of elements generated across all rows is $\sum_{i=1}^{numRows} i = \frac{numRows \times (numRows + 1)}{2}$. Each element is calculated in $\mathcal{O}(1)$ time. For $numRows = 30$, this is only $465$ operations.
- **Space Complexity:** $\mathcal{O}(numRows^2)$ required to store and return the full triangular structure. Beyond the returned output, auxiliary memory is $\mathcal{O}(1)$.

---

### Takeaway Pattern & Interview Traps

1. **Pre-filling with $1$s:** Initializing each row as `[1] * (i + 1)` automatically sets the boundary values `row[0] = 1` and `row[i] = 1`, completely avoiding special boundary checks in the inner loop.
2. **0-indexed vs. 1-indexed:** Be clear on whether row counting starts from 0 or 1. Here, $numRows$ rows means indices $0$ to $numRows - 1$.
3. **Contrast with LC 119:** LeetCode 119 asks for only the $k^{\text{th}}$ row, which can be solved in $\mathcal{O}(k)$ space in-place using a single 1D array traversed backwards.