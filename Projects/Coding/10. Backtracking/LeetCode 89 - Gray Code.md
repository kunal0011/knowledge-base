---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 89: Gray Code"
tags:
  - leetcode
  - coding
  - backtracking
  - bit-manipulation
  - math
  - amazon
  - google
---

# LeetCode 89: Gray Code

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Backtracking / Bit Manipulation / Hypercube Traversal  

---

### Problem Statement

An **n-bit gray code sequence** is a sequence of $2^n$ integers where:
- Every integer is in the inclusive range $[0, 2^n - 1]$.
- The first integer is $0$.
- An integer appears **no more than once** in the sequence.
- The binary representation of every pair of adjacent integers differs by **exactly one bit**.
- The binary representation of the first and last integers differs by **exactly one bit**.

Given an integer `n`, return *any valid n-bit gray code sequence*.

---

### Input & Output Formats & Constraints

- **Input:** `n: int`
- **Output:** `List[int]` containing $2^n$ integers representing the Gray code sequence.
- **Constraints:**
  - $1 \le n \le 16$

---

### Key Idea & Intuition

- **The Hypercube View (Hamiltonian Cycle):**
  - An $n$-bit binary string corresponds to a vertex in an $n$-dimensional hypercube.
  - Two vertices share an undirected edge if their Hamming distance is $1$ (they differ by a single bit flip).
  - A Gray code is simply a **Hamiltonian path/cycle** that starts at vertex $0$ and visits all $2^n$ vertices exactly once.
- **Approach 1: Direct Bitwise Formula ($\mathcal{O}(2^n)$ Time, $\mathcal{O}(1)$ Extra Space):**
  - There is a closed-form formula for the $i$-th Gray code integer:
    $$G(i) = i \oplus (i \gg 1)$$
  - For $i = 0, 1, \dots, 2^n - 1$, computing $i \oplus (i \gg 1)$ directly generates the entire Gray code sequence in strictly linear time relative to output size!
- **Approach 2: Recursive Mirror Reflection:**
  - Base case: For $n = 0$, sequence is `[0]`.
  - To generate the sequence for $n = k$ from $n = k - 1$:
    - Take the $(k - 1)$-bit sequence.
    - Reverse it and add $2^{k-1}$ (setting the $k$-th bit) to each element.
    - Concatenate the two halves!
    - Example for $n=1 \to n=2$:
      - $n=1$: `[0, 1]`
      - Reverse + set bit: `[1 + 2, 0 + 2] = [3, 2]`
      - Combined $n=2$: `[0, 1, 3, 2]`!

---

### Solution Approach (Step-by-Step)

#### Direct Formula Generation (Optimal):
1. Initialize `results` array of capacity $2^n$.
2. For $i$ from $0$ to $2^n - 1$:
   - Append `i ^ (i >> 1)` to `results`.
3. Return `results`.

#### Mirroring Algorithm:
1. Initialize `res = [0]`.
2. For $i$ from $0$ to $n - 1$:
   - Current bit mask: `mask = 1 << i`.
   - Iterate through `res` backwards from `len(res) - 1` down to $0$:
     - Append `res[j] | mask` to `res`.
3. Return `res`.

---

### Visual Algorithm Walkthrough

Mirroring progression from $n=1$ to $n=3$:

```
n = 1:
  0
  1
----------------- (Mirror line)
n = 2:
  00
  01
  -- (add bit 2: 1 << 1 = 2)
  11 (1 + 2)
  10 (0 + 2)
----------------- (Mirror line)
n = 3:
  000
  001
  011
  010
  --- (add bit 3: 1 << 2 = 4 to reversed prefix)
  110 (2 + 4)
  111 (3 + 4)
  101 (1 + 4)
  100 (0 + 4)

Result for n=3: [0, 1, 3, 2, 6, 7, 5, 4]
Notice that 0 ("000") and 4 ("100") also differ by exactly 1 bit (cyclic).
```

---

### Solved Examples with Multiple Inputs

| Test Case | `n` | Size $2^n$ | Binary Representation | Integer Sequence |
| :--- | :--- | :--- | :--- | :--- |
| **n = 1** | `1` | `2` | `0, 1` | `[0, 1]` |
| **n = 2** | `2` | `4` | `00, 01, 11, 10` | `[0, 1, 3, 2]` |
| **n = 3** | `3` | `8` | `000, 001, 011, 010, 110, 111, 101, 100` | `[0, 1, 3, 2, 6, 7, 5, 4]` |
| **n = 4** | `4` | `16` | 16 cyclic single-bit changes | 16 elements |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def grayCode(self, n: int) -> List[int]:
        """
        Generates an n-bit Gray code sequence using the closed-form bitwise formula.
        O(2^n) time and O(1) auxiliary space.
        """
        total = 1 << n
        return [i ^ (i >> 1) for i in range(total)]

    def grayCodeMirror(self, n: int) -> List[int]:
        """
        Alternative: Mirror reflection method.
        """
        res = [0]
        for i in range(n):
            mask = 1 << i
            # Append mirrored elements with the new highest bit set
            for j in range(len(res) - 1, -1, -1):
                res.append(res[j] | mask)
        return res
```

#### C++17
```cpp
#include <vector>

class Solution {
public:
    std::vector<int> grayCode(int n) {
        int total = 1 << n;
        std::vector<int> results;
        results.reserve(total);

        for (int i = 0; i < total; ++i) {
            // Direct Gray code mapping: i ^ (i >> 1)
            results.push_back(i ^ (i >> 1));
        }

        return results;
    }
};
```

#### Java
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<Integer> grayCode(int n) {
        int total = 1 << n;
        List<Integer> results = new ArrayList<>(total);

        for (int i = 0; i < total; i++) {
            // Direct mathematical conversion
            results.add(i ^ (i >> 1));
        }

        return results;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(2^n)$.
  - The loop runs exactly $2^n$ times.
  - Each step computes $i \oplus (i \gg 1)$ in $\mathcal{O}(1)$ CPU operations.
  - For $n \le 16$, $2^{16} = 65,536$ operations $\implies < 2 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space beyond the output array of length $2^n$.

---

### Takeaway Pattern & Interview Traps

- **The Closed Form `i ^ (i >> 1)`:** Memorizing this formula saves 30 minutes of writing and debugging recursive backtracking logic in interviews.
- **Mirror Reflection Invariant:** The reflection property ensures that adjacent codes differ by only the newly added highest bit at the midpoint, and by only 1 bit between the last and first elements, ensuring the cyclic property.