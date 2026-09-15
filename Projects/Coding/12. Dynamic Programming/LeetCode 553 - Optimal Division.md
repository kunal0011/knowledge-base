---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 553: Optimal Division"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - math
  - greedy
  - string
  - amazon
  - microsoft
---

# LeetCode 553: Optimal Division

**Target Companies:** Amazon, Microsoft, Google  
**Difficulty:** Medium  
**Topic:** Mathematical Proof / Greedy String Formatting / Interval Dynamic Programming  

---

### Problem Statement

You are given an integer array `nums`. The adjacent integers in `nums` will perform the float division.

- For example, if `nums = [2, 3, 4]`, we will evaluate the expression `"2/3/4"`.

However, you can add any number of parentheses at any position to change the priority of operations. You want to add these parentheses such that the value of the expression after the evaluation is **maximized**.

Return the corresponding expression that has the maximum value in string format.

**Note:** Your expression should not contain redundant parentheses.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]` — Array of integers where each integer is $\ge 2$.
- **Output:** `str` — Optimal parenthesized expression maximizing the division result.
- **Constraints:**
  - $1 \le \text{nums.length} \le 10$
  - $2 \le \text{nums}[i] \le 1000$
  - There is only one unique configuration that achieves the maximum value.

---

### Key Idea & Intuition

1. **Mathematical Invariant:**
   - Any valid parenthesization of $[X_0, X_1, X_2, \dots, X_{n-1}]$ under real division evaluates to a fraction of the form:
     $$\frac{\prod_{i \in \text{Numerator}} X_i}{\prod_{j \in \text{Denominator}} X_j}$$
   - Notice two fundamental mathematical constants of this expression:
     1. $X_0$ is **always in the numerator**, because no operator precedes it.
     2. $X_1$ is **always in the denominator**, because the first division $X_0 / \dots$ always places $X_1$ below the primary fraction bar.
     3. For every subsequent number $X_2, X_3, \dots, X_{n-1}$, we can choose whether it lands in the numerator or denominator depending on parentheses.
   - Because all numbers in `nums` are strictly greater than $1$ ($X_i \ge 2$), to **maximize** the final result, we should place **every single number from $X_2$ to $X_{n-1}$ into the numerator**!

2. **The Universal Parenthesization Pattern:**
   - How can we force all elements $X_2, \dots, X_{n-1}$ into the numerator?
   - Wrap the entire suffix from $X_1$ to $X_{n-1}$ in a single set of parentheses:
     $$X_0 / (X_1 / X_2 / X_3 / \dots / X_{n-1})$$
   - By algebra:
     $$X_0 / \left(\frac{X_1}{X_2 \times X_3 \times \dots \times X_{n-1}}\right) = \frac{X_0 \times X_2 \times X_3 \times \dots \times X_{n-1}}{X_1}$$
   - This achieves the absolute theoretical maximum possible value of the expression in $\mathcal{O}(N)$ time!

3. **Interval DP Perspective (Min-Max Recurrence):**
   - The classical DP formulation defines:
     $$\text{dp}[i][j] = (\text{max\_val}, \text{min\_val})$$
     $$\text{max\_val} = \max_{i \le k < j} \frac{\text{dp}[i][k].\text{max\_val}}{\text{dp}[k+1][j].\text{min\_val}}$$
     $$\text{min\_val} = \min_{i \le k < j} \frac{\text{dp}[i][k].\text{min\_val}}{\text{dp}[k+1][j].\text{max\_val}}$$
   - Solving this interval DP confirms the exact same mathematical conclusion derived above.

---

### Solution Approach (Step-by-Step)

1. **Check Base Cases:**
   - If $n = 1$: return `str(nums[0])`.
   - If $n = 2$: return `f"{nums[0]}/{nums[1]}"`.
2. **Format Suffix Parentheses ($n \ge 3$):**
   - Build string: `f"{nums[0]}/({nums[1]}/{nums[2]}/.../{nums[n-1]})"`.
3. **Return:**
   - Return the formatted string.

---

### Visual Algorithm Walkthrough

For `nums = [1000, 100, 10, 2]`:

```
Without Parentheses:
  1000 / 100 / 10 / 2 = ((1000 / 100) / 10) / 2 = (10 / 10) / 2 = 1 / 2 = 0.5

With Optimal Suffix Parentheses:
  1000 / (100 / 10 / 2)
  Inner expression: 100 / 10 / 2 = (100 / 10) / 2 = 10 / 2 = 5
  Outer expression: 1000 / 5 = 200

Algebraic Form:
  1000 * 10 * 2 / 100 = 20000 / 100 = 200 (Maximum Possible!)

Output String: "1000/(100/10/2)"
```

---

### Solved Examples with Multiple Inputs

| Case | `nums` | Formatted Expression | Evaluated Value | Explanation |
|---|---|---|---|---|
| **Standard 4** | `[1000, 100, 10, 2]` | `"1000/(100/10/2)"` | `200.0` | Maximizes numerator factors |
| **Two Elements** | `[2, 3]` | `"2/3"` | `0.666...` | No parentheses needed |
| **Single Element** | `[5]` | `"5"` | `5.0` | No division operators |
| **Three Elements** | `[10, 2, 5]` | `"10/(2/5)"` | $10 / 0.4 = 25.0$ | $10 \times 5 / 2 = 25$ |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def optimalDivision(self, nums: List[int]) -> str:
        n = len(nums)
        if n == 1:
            return str(nums[0])
        if n == 2:
            return f"{nums[0]}/{nums[1]}"
            
        # Group everything after nums[0] into a single denominator block
        suffix = "/".join(str(x) for x in nums[1:])
        return f"{nums[0]}/({suffix})"
```

#### 2. C++ (C++17 / STL)
```cpp
#include <string>
#include <vector>

class Solution {
public:
    std::string optimalDivision(std::vector<int>& nums) {
        int n = nums.size();
        if (n == 1) return std::to_string(nums[0]);
        if (n == 2) return std::to_string(nums[0]) + "/" + std::to_string(nums[1]);

        std::string result = std::to_string(nums[0]) + "/(" + std::to_string(nums[1]);
        for (int i = 2; i < n; ++i) {
            result += "/" + std::to_string(nums[i]);
        }
        result += ")";

        return result;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public String optimalDivision(int[] nums) {
        int n = nums.length;
        if (n == 1) return String.valueOf(nums[0]);
        if (n == 2) return nums[0] + "/" + nums[1];

        StringBuilder sb = new StringBuilder();
        sb.append(nums[0]).append("/(").append(nums[1]);
        for (int i = 2; i < n; i++) {
            sb.append("/").append(nums[i]);
        }
        sb.append(")");

        return sb.toString();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$  
  Constructing the output string takes a single pass over $N$ numbers. For $N \le 10$, this runs in $< 0.1$ ms.
- **Space Complexity:** $\mathcal{O}(N)$  
  Only the output string buffer of length proportional to $N$ is created.

---

### Takeaway Pattern & Interview Traps

1. **The Math Insight Saves $\mathcal{O}(N^3)$ Interval DP:**
   - Although the problem appears to require $\mathcal{O}(N^3)$ interval dynamic programming (similar to Matrix Chain Multiplication), recognizing that all elements are $\ge 2$ collapses the problem into an $\mathcal{O}(N)$ greedy mathematical pattern.
2. **Base Cases Handling:**
   - Always check $n = 1$ and $n = 2$ separately. Putting parentheses around a single element (e.g. `100/(2)`) violates the "no redundant parentheses" rule.