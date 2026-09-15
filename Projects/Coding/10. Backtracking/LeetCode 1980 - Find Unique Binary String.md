---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 1980: Find Unique Binary String"
tags:
  - leetcode
  - coding
  - backtracking
  - math
  - string
  - amazon
  - google
---

# LeetCode 1980: Find Unique Binary String

**Target Companies:** Amazon, Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Backtracking / State-Space Search / Cantor's Diagonal Argument  

---

### Problem Statement

Given an array of strings `nums` containing `n` **unique** binary strings each of length `n`, return *a binary string of length `n` that **does not appear** in `nums`*. If there are multiple answers, you may return **any of them**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[str]` where `len(nums) == n` and `len(nums[i]) == n`.
- **Output:** `str` (any binary string of length `n` absent from `nums`).
- **Constraints:**
  - $n == \text{nums.length}$
  - $1 \le n \le 16$
  - $\text{nums}[i]\text{.length} == n$
  - `nums[i]` consists of only `'0'` or `'1'`.
  - All strings in `nums` are **unique**.

---

### Key Idea & Intuition

- **The Mathematical Reality:**
  - There are $2^n$ possible binary strings of length $n$.
  - The input contains only $n$ strings.
  - Since $2^n > n$ for all $n \ge 1$ (e.g. for $n = 16$, $2^{16} = 65,536 > 16$), the vast majority of binary strings are missing from `nums`.

- **Approach 1: Backtracking (State-Space DFS):**
  - Store `nums` in a hash set for $\mathcal{O}(1)$ lookup.
  - Explore the binary decision tree of depth $n$: at each index, try `'0'` then `'1'`.
  - Once depth $n$ is reached, if the generated string is not in the set, immediately return it up the call stack. Because at most $n + 1$ leaves need to be visited before finding a missing string (by Pigeonhole Principle), this search terminates almost instantly.

- **Approach 2: Cantor's Diagonal Argument ($\mathcal{O}(n)$ Time & $\mathcal{O}(1)$ Auxiliary Space):**
  - Form a string `ans` of length $n$ where the $i$-th character is the complement of the $i$-th character of `nums[i]`:
    $$\text{ans}[i] = \begin{cases} \text{'1'} & \text{if } \text{nums}[i][i] == \text{'0'} \\ \text{'0'} & \text{if } \text{nums}[i][i] == \text{'1'} \end{cases}$$
  - **Why this works:** `ans` differs from `nums[0]` at index 0, from `nums[1]` at index 1, $\dots$, and from `nums[n-1]` at index $n - 1$. Therefore, `ans` cannot equal any string in `nums`!

---

### Solution Approach (Step-by-Step)

#### Method 1: Backtracking DFS
1. Insert all strings from `nums` into a hash set `seen`.
2. Define `dfs(curr_path)`:
   - If `len(curr_path) == n`:
     - If `curr_path not in seen`: return `curr_path`.
     - Else: return `""`.
   - Try appending `'0'`:
     - `res = dfs(curr_path + "0")`
     - If `res != ""`: return `res`.
   - Try appending `'1'`:
     - `res = dfs(curr_path + "1")`
     - If `res != ""`: return `res`.
   - Return `""`.

#### Method 2: Cantor's Diagonalization
1. Initialize an empty list or string builder `res`.
2. For $i$ from $0$ to $n - 1$:
   - If `nums[i][i] == '0'`, append `'1'`.
   - Else, append `'0'`.
3. Return `"".join(res)`.

---

### Visual Algorithm Walkthrough

#### Cantor's Diagonalization Method:
Let $n = 3$, `nums = ["011", "101", "000"]`:

```
Matrix of strings:
  Index: 0  1  2
nums[0]: [0] 1  1   -> invert nums[0][0] ('0') => '1'
nums[1]:  1 [0] 1   -> invert nums[1][1] ('0') => '1'
nums[2]:  0  0 [0]  -> invert nums[2][2] ('0') => '1'

Resulting string: "111"

Verification:
- Differs from nums[0] ("011") at index 0: '1' vs '0'  ✓
- Differs from nums[1] ("101") at index 1: '1' vs '0'  ✓
- Differs from nums[2] ("000") at index 2: '1' vs '0'  ✓
"111" is guaranteed not in nums!
```

---

### Solved Examples with Multiple Inputs

| Test Case | `nums` | Missing Candidates | Cantor's Diagonal Result | Valid? |
| :--- | :--- | :--- | :--- | :--- |
| **Example 1** | `["01","10"]` | `"00"`, `"11"` | `nums[0][0]='0'->'1'`, `nums[1][1]='0'->'1'` $\implies$ `"11"` | Yes |
| **Example 2** | `["00","01"]` | `"10"`, `"11"` | `nums[0][0]='0'->'1'`, `nums[1][1]='1'->'0'` $\implies$ `"10"` | Yes |
| **Example 3** | `["111","011","001"]` | `"000","010","100","101","110"` | Invert diagonals: `1,1,1` $\implies$ `"000"` | Yes |
| **Single Element** | `["0"]` | `"1"` | Invert `nums[0][0]='0'->'1'` $\implies$ `"1"` | Yes |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        """
        Generates a unique binary string of length n absent from nums.
        Demonstrates Cantor's Diagonalization in O(n) time.
        """
        n = len(nums)
        # Invert the i-th character of the i-th string
        return "".join('1' if nums[i][i] == '0' else '0' for i in range(n))

    def findDifferentBinaryStringBacktracking(self, nums: List[str]) -> str:
        """
        Alternative: Backtracking state-space search.
        Visits at most n + 1 leaves by the Pigeonhole Principle.
        """
        n = len(nums)
        seen = set(nums)
        path = []

        def dfs(depth: int) -> str:
            if depth == n:
                candidate = "".join(path)
                return candidate if candidate not in seen else ""

            for bit in ('0', '1'):
                path.append(bit)
                res = dfs(depth + 1)
                if res:
                    return res
                path.pop()  # Backtrack

            return ""

        return dfs(0)
```

#### C++17
```cpp
#include <string>
#include <vector>

class Solution {
public:
    // Approach 1: Cantor's Diagonal Argument - O(n) time, O(1) auxiliary space
    std::string findDifferentBinaryString(const std::vector<std::string>& nums) {
        int n = static_cast<int>(nums.size());
        std::string result = "";
        result.reserve(n);

        for (int i = 0; i < n; ++i) {
            // Invert the diagonal bit
            result.push_back(nums[i][i] == '0' ? '1' : '0');
        }

        return result;
    }
};
```

#### Java
```java
class Solution {
    // Approach 1: Cantor's Diagonal Argument - O(n) time, O(1) auxiliary space
    public String findDifferentBinaryString(String[] nums) {
        int n = nums.length;
        StringBuilder sb = new StringBuilder(n);

        for (int i = 0; i < n; i++) {
            // Invert the diagonal bit at index i
            sb.append(nums[i].charAt(i) == '0' ? '1' : '0');
        }

        return sb.toString();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - **Cantor's Diagonalization:** $\mathcal{O}(n)$ because we inspect exactly 1 character for each of the $n$ strings and construct a string of length $n$.
  - **Backtracking:** $\mathcal{O}(n^2)$ worst-case. By the Pigeonhole Principle, among the first $n + 1$ binary strings generated, at least one must not belong to `nums`. Checking each against a hash set takes $\mathcal{O}(n)$.
- **Space Complexity:**
  - **Cantor's Diagonalization:** $\mathcal{O}(1)$ auxiliary space beyond the output string of length $n$.
  - **Backtracking:** $\mathcal{O}(n)$ auxiliary space for the recursion call stack and current path buffer (plus $\mathcal{O}(n^2)$ for the hash set).

---

### Takeaway Pattern & Interview Traps

- **Cantor's Diagonal Trick:** Whenever asked to construct a sequence of length $N$ that differs from $N$ given sequences, invert the $i$-th element of the $i$-th sequence. It is the most elegant, optimal mathematical trick in coding interviews.
- **Pigeonhole Principle in Backtracking:** If using backtracking, note that you do **not** need to search all $2^n$ combinations. Visiting just the first $n + 1$ leaves guarantees discovering an absent string.