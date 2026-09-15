---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 241: Different Ways to Add Parentheses"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - divide-and-conquer
  - memoization
  - string
  - amazon
  - google
  - microsoft
---

# LeetCode 241: Different Ways to Add Parentheses

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Divide and Conquer / Memoization  

---

### Problem Statement

Given a string `expression` of numbers and operators, return *all possible results from computing all the different possible ways to group numbers and operators*. You may return the answer in **any order**.

The test cases are generated such that the output values fit in a 32-bit integer and the number of different results does not exceed $10^4$.

---

### Input & Output Formats & Constraints

- **Input:** A string `expression` ($1 \le |expression| \le 20$) containing digits and operators `'+'`, `'-'`, `'*'`.
- **Output:** A list of integers `List[int]` representing all possible computed results.
- **Constraints:**
  - `1 <= expression.length <= 20`
  - `expression` consists of digits and the operator `'+'`, `'-'`, and `'*'`.
  - All the integer values in the input expression are in the range `[0, 99]`.

---

### Key Idea & Intuition

#### Divide and Conquer Over Operators
Every grouping of parentheses corresponds to picking one operator as the **final operation** executed:
1. If we choose the operator at index $i$ (`'+'`, `'-'`, or `'*'`) to be the final operation:
   - The expression is partitioned into two independent subexpressions:
     - Left subexpression: `expression[0..i-1]`
     - Right subexpression: `expression[i+1..end]`
2. Recursively, `expression[0..i-1]` yields a list of possible numbers $L$, and `expression[i+1..end]` yields a list of possible numbers $R$.
3. We compute the Cartesian product $L \times R$: for every $a \in L$ and $b \in R$, evaluate $a \text{ op } b$.

#### Overlapping Subproblems & Memoization
Different evaluation trees share identical subexpressions (e.g. `2 * 3` appears repeatedly in `(2 * 3) - (4 * 5)` and `((2 * 3) - 4) * 5`).
Caching the results of each unique substring in a hash map `memo[expression]` avoids recalculating the combinatorial tree repeatedly.

---

### Solution Approach (Step-by-Step)

1. **Memoization Map:**
   - Maintain a dictionary/map `memo: string -> List[int]`.
2. **Recursive Function `compute(expr)`:**
   - If `expr` in `memo`, return `memo[expr]`.
   - Initialize `results = []`.
   - For each character at index $i$ in `expr`:
     - If `expr[i]` is an operator (`'+'`, `'-'`, `'*'`):
       - `left_vals = compute(expr[:i])`
       - `right_vals = compute(expr[i+1:])`
       - For $a$ in `left_vals`:
         - For $b$ in `right_vals`:
           - If `expr[i] == '+'`: `results.append(a + b)`
           - If `expr[i] == '-'`: `results.append(a - b)`
           - If `expr[i] == '*'`: `results.append(a * b)`
   - **Base Case (Pure Number):**
     - If `results` is empty (meaning no operator was found in `expr`), `expr` is a single integer: `results.append(int(expr))`.
   - Save `memo[expr] = results` and return.
3. **Return:**
   - Call `compute(expression)`.

---

### Visual Algorithm Walkthrough

#### Trace for `expression = "2*3-4*5"`
```
Divide and Conquer on Operators:

Split 1 at '*': left = "2", right = "3-4*5"
  - "2" -> [2]
  - "3-4*5":
      Split at '-': "3" and "4*5" -> 3 - 20 = -17
      Split at '*': "3-4" and "5" -> -1 * 5 = -5
      -> returns [-17, -5]
  Combinations:
  2 * (-17) = -34
  2 * (-5) = -10

Split 2 at '-': left = "2*3", right = "4*5"
  - "2*3" -> [6]
  - "4*5" -> [20]
  Combinations:
  6 - 20 = -14

Split 3 at '*': left = "2*3-4", right = "5"
  - "2*3-4" -> [-2, 2]
  - "5" -> [5]
  Combinations:
  -2 * 5 = -10
  2 * 5 = 10

Combined Results: [-34, -10, -14, -10, 10].
```

---

### Solved Examples with Multiple Inputs

| `expression` | Number of Operators | Possible Evaluation Groupings | Output |
|---|---|---|---|
| `"2-1-1"` | 2 | `((2-1)-1) = 0`, `(2-(1-1)) = 2` | `[0, 2]` |
| `"2*3-4*5"` | 3 | Full trace shown above | `[-34, -14, -10, -10, 10]` |
| `"42"` | 0 | Single number | `[42]` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def diffWaysToCompute(self, expression: str) -> list[int]:
        memo: dict[str, list[int]] = {}
        
        def compute(expr: str) -> list[int]:
            if expr in memo:
                return memo[expr]
                
            res: list[int] = []
            for i, ch in enumerate(expr):
                if ch in "+-*":
                    left_vals = compute(expr[:i])
                    right_vals = compute(expr[i + 1:])
                    
                    for a in left_vals:
                        for b in right_vals:
                            if ch == '+':
                                res.append(a + b)
                            elif ch == '-':
                                res.append(a - b)
                            elif ch == '*':
                                res.append(a * b)
                                
            # Base case: pure number with no operators
            if not res:
                res.append(int(expr))
                
            memo[expr] = res
            return res
            
        return compute(expression)
```

#### C++17
```cpp
#include <vector>
#include <string>
#include <unordered_map>

class Solution {
private:
    std::unordered_map<std::string, std::vector<int>> memo;

public:
    std::vector<int> diffWaysToCompute(const std::string& expression) {
        if (memo.count(expression)) {
            return memo[expression];
        }

        std::vector<int> results;
        int n = static_cast<int>(expression.size());

        for (int i = 0; i < n; ++i) {
            char ch = expression[i];
            if (ch == '+' || ch == '-' || ch == '*') {
                std::vector<int> left = diffWaysToCompute(expression.substr(0, i));
                std::vector<int> right = diffWaysToCompute(expression.substr(i + 1));

                for (int a : left) {
                    for (int b : right) {
                        if (ch == '+') results.push_back(a + b);
                        else if (ch == '-') results.push_back(a - b);
                        else if (ch == '*') results.push_back(a * b);
                    }
                }
            }
        }

        if (results.empty()) {
            results.push_back(std::stoi(expression));
        }

        memo[expression] = results;
        return results;
    }
};
```

#### Java 17
```java
import java.util.*;

class Solution {
    private Map<String, List<Integer>> memo = new HashMap<>();

    public List<Integer> diffWaysToCompute(String expression) {
        if (memo.containsKey(expression)) {
            return memo.get(expression);
        }

        List<Integer> results = new ArrayList<>();
        int n = expression.length();

        for (int i = 0; i < n; i++) {
            char ch = expression.charAt(i);
            if (ch == '+' || ch == '-' || ch == '*') {
                List<Integer> left = diffWaysToCompute(expression.substring(0, i));
                List<Integer> right = diffWaysToCompute(expression.substring(i + 1));

                for (int a : left) {
                    for (int b : right) {
                        if (ch == '+') results.add(a + b);
                        else if (ch == '-') results.add(a - b);
                        else if (ch == '*') results.add(a * b);
                    }
                }
            }
        }

        if (results.isEmpty()) {
            results.add(Integer.parseInt(expression));
        }

        memo.put(expression, results);
        return results;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(C_n \cdot n)$, where $n$ is the number of operators and $C_n = \frac{1}{n+1}\binom{2n}{n}$ is the $n^{\text{th}}$ Catalan number (representing full binary trees with $n$ internal nodes). For $n \le 10$, $C_{10} = 16796$, running in under $5$ ms.
- **Space Complexity:** $\mathcal{O}(C_n)$ to store all intermediate expression evaluations in the memoization table.

---

### Takeaway Pattern & Interview Traps

1. **Catalan Tree Structure:** The number of ways to add parentheses to an expression with $n$ operators is isomorphic to the number of structurally unique Full Binary Trees with $n$ internal operator nodes.
2. **Duplicate Values Allowed:** Note that the output can contain duplicate numbers (e.g. `-10` appears twice in `2*3-4*5` from different evaluation trees). The problem asks for all different evaluation ways, not distinct values; do not use a set to deduplicate.
3. **Relation to Interval DP:** This problem is equivalent to Matrix Chain Multiplication and Burst Balloons (LC 312), where an operator split at $k$ divides $[i, j]$ into $[i, k]$ and $[k+1, j]$.