---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 131: Palindrome Partitioning"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - backtracking
  - string
  - amazon
  - google
  - meta
  - microsoft
---

# LeetCode 131: Palindrome Partitioning

**Target Companies:** Amazon, Google, Meta, Microsoft, Bloomberg, Apple  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Backtracking / String  

---

### Problem Statement

Given a string `s`, partition `s` such that every substring of the partition is a **palindrome**. Return *all possible palindrome partitioning of `s`*.

---

### Input & Output Formats & Constraints

- **Input:** A string `s` ($1 \le |s| \le 16$) consisting of lowercase English letters.
- **Output:** A list of lists of strings `List[List[str]]` containing all valid partition sets.
- **Constraints:**
  - `1 <= s.length <= 16`
  - `s` contains only lowercase English letters.

---

### Key Idea & Intuition

#### Two-Layer Architecture: DP + Backtracking
This problem combines two algorithmic paradigms:
1. **Layer 1: Dynamic Programming for $\mathcal{O}(1)$ Palindrome Queries:**
   - A naive palindrome check takes $\mathcal{O}(L)$ time per substring.
   - We can precompute a 2D boolean table $dp[i][j]$ indicating whether substring $s[i..j]$ is a palindrome:
     $$dp[i][j] = (s[i] == s[j]) \land (j - i \le 2 \lor dp[i + 1][j - 1])$$
     - If length $\le 2$ ($j - i \le 1$), matching outer characters $s[i] == s[j]$ directly implies a palindrome.
     - For length $\ge 3$, outer characters must match AND the inner substring $s[i+1..j-1]$ must be a palindrome.
2. **Layer 2: Backtracking DFS for Combinatorial Partitioning:**
   - With the DP table populated, a backtracking function `dfs(start, path)` explores all cuts.
   - For each index $end \in [start, n - 1]$:
     - If $dp[start][end]$ is `true`:
       - Append $s[start..end]$ to `path`.
       - Recurse on `dfs(end + 1, path)`.
       - Backtrack (pop from `path`).
   - Base Case: When $start == n$, the entire string has been successfully partitioned into palindromes; record `path`.

---

### Solution Approach (Step-by-Step)

1. **Precompute Palindrome DP Table:**
   - Let $n = |s|$.
   - Create a 2D boolean array `dp` of size $n \times n$ initialized to `false`.
   - Traverse $i$ from $n - 1$ down to 0:
     - For $j$ from $i$ to $n - 1$:
       - If $s[i] == s[j]$ and $(j - i \le 2 \text{ or } dp[i + 1][j - 1])$:
         - $dp[i][j] = \text{true}$.
2. **Backtracking DFS:**
   - Define recursive helper `dfs(start, current_path)`:
     - If $start == n$:
       - Append a copy of `current_path` to `result`.
       - Return.
     - For $end$ from $start$ to $n - 1$:
       - If $dp[start][end]$:
         - `current_path.append(s[start:end+1])`
         - `dfs(end + 1, current_path)`
         - `current_path.pop()`
3. **Return Output:**
   - Invoke `dfs(0, [])` and return `result`.

---

### Visual Algorithm Walkthrough

#### Trace for `s = "aab"`
```
Step 1: DP Palindrome Table Construction:
  i = 2 ('b'):
    j = 2 ('b'): s[2]==s[2] -> dp[2][2] = True ("b")
  i = 1 ('a'):
    j = 1 ('a'): s[1]==s[1] -> dp[1][1] = True ("a")
    j = 2 ('b'): s[1]!=s[2] -> dp[1][2] = False ("ab")
  i = 0 ('a'):
    j = 0 ('a'): s[0]==s[0] -> dp[0][0] = True ("a")
    j = 1 ('a'): s[0]==s[1] -> dp[0][1] = True ("aa")
    j = 2 ('b'): s[0]!=s[2] -> dp[0][2] = False ("aab")

DP Palindrome Table:
     0(a)  1(a)  2(b)
0(a)  T     T     F
1(a)  .     T     F
2(b)  .     .     T

Step 2: Backtracking Tree from start = 0:
dfs(0, [])
 ├── end=0: dp[0][0] is True ("a")
 │    dfs(1, ["a"])
 │     ├── end=1: dp[1][1] is True ("a")
 │     │    dfs(2, ["a", "a"])
 │     │     └── end=2: dp[2][2] is True ("b")
 │     │          dfs(3, ["a", "a", "b"]) -> start==3 -> Collect ["a", "a", "b"]
 │     └── end=2: dp[1][2] is False ("ab") -> Pruned
 └── end=1: dp[0][1] is True ("aa")
      dfs(2, ["aa"])
       └── end=2: dp[2][2] is True ("b")
            dfs(3, ["aa", "b"]) -> start==3 -> Collect ["aa", "b"]

Output: [["a", "a", "b"], ["aa", "b"]]
```

---

### Solved Examples with Multiple Inputs

| String `s` | Palindromic Substrings Found | Backtracking Branching Paths | Output Partitions |
|---|---|---|---|
| `"aab"` | `"a"`, `"aa"`, `"b"` | `["a", "a", "b"]`, `["aa", "b"]` | `[["a","a","b"], ["aa","b"]]` |
| `"a"` | `"a"` | `["a"]` | `[["a"]]` |
| `"efe"` | `"e"`, `"f"`, `"efe"` | `["e", "f", "e"]`, `["efe"]` | `[["e","f","e"], ["efe"]]` |
| `"racecar"` | `"r"`,`"a"`,`"c"`,`"e"`,`"cec"`,`"aceca"`,`"racecar"` | Multiple symmetrical paths | Includes `["racecar"]`, `["r","aceca","r"]`, ... |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def partition(self, s: str) -> list[list[str]]:
        n = len(s)
        
        # Layer 1: DP precomputation for palindrome checks
        dp = [[False] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i <= 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                    
        result: list[list[str]] = []
        
        # Layer 2: Backtracking to construct partitions
        def dfs(start: int, current_path: list[str]) -> None:
            if start == n:
                result.append(list(current_path))
                return
                
            for end in range(start, n):
                if dp[start][end]:
                    current_path.append(s[start:end + 1])
                    dfs(end + 1, current_path)
                    current_path.pop()
                    
        dfs(0, [])
        return result
```

#### C++17
```cpp
#include <vector>
#include <string>

class Solution {
private:
    void dfs(int start, const std::string& s, const std::vector<std::vector<bool>>& dp,
             std::vector<std::string>& path, std::vector<std::vector<std::string>>& result) {
        int n = static_cast<int>(s.size());
        if (start == n) {
            result.push_back(path);
            return;
        }

        for (int end = start; end < n; ++end) {
            if (dp[start][end]) {
                path.push_back(s.substr(start, end - start + 1));
                dfs(end + 1, s, dp, path, result);
                path.pop_back();
            }
        }
    }

public:
    std::vector<std::vector<std::string>> partition(const std::string& s) {
        int n = static_cast<int>(s.size());
        std::vector<std::vector<bool>> dp(n, std::vector<bool>(n, false));

        // Precompute palindrome lookup table
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i; j < n; ++j) {
                if (s[i] == s[j] && (j - i <= 2 || dp[i + 1][j - 1])) {
                    dp[i][j] = true;
                }
            }
        }

        std::vector<std::vector<std::string>> result;
        std::vector<std::string> path;
        dfs(0, s, dp, path, result);
        return result;
    }
};
```

#### Java 17
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    private void dfs(int start, String s, boolean[][] dp, List<String> path, List<List<String>> result) {
        int n = s.length();
        if (start == n) {
            result.add(new ArrayList<>(path));
            return;
        }

        for (int end = start; end < n; end++) {
            if (dp[start][end]) {
                path.add(s.substring(start, end + 1));
                dfs(end + 1, s, dp, path, result);
                path.remove(path.size() - 1);
            }
        }
    }

    public List<List<String>> partition(String s) {
        int n = s.length();
        boolean[][] dp = new boolean[n][n];

        // Bottom-up DP to identify all palindromic substrings
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (s.charAt(i) == s.charAt(j) && (j - i <= 2 || dp[i + 1][j - 1])) {
                    dp[i][j] = true;
                }
            }
        }

        List<List<String>> result = new ArrayList<>();
        List<String> path = new ArrayList<>();
        dfs(0, s, dp, path, result);
        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n \cdot 2^n)$. Populating the DP palindrome table takes $\mathcal{O}(n^2)$ time. In the worst case (e.g. `s = "aaaa..."`), there are $2^{n-1}$ valid partitions, and constructing/copying each partition of strings takes $\mathcal{O}(n)$ time. Total runtime is $\mathcal{O}(n^2 + n \cdot 2^n) = \mathcal{O}(n \cdot 2^n)$. For $n \le 16$, $16 \cdot 2^{16} \approx 10^6$ operations, executing within $10$ ms.
- **Space Complexity:** $\mathcal{O}(n^2)$ for the 2D palindrome DP table, plus $\mathcal{O}(n)$ recursion depth for the DFS call stack.

---

### Takeaway Pattern & Interview Traps

1. **Why Backtracking is Unavoidable:** Because the problem requires generating *all* combinations of substrings, the output size itself is $\mathcal{O}(2^n)$ in the worst case. Dynamic Programming cannot reduce the combinatorial output size, but it optimizes substring checking from $\mathcal{O}(n)$ to $\mathcal{O}(1)$.
2. **Fill Order for Palindrome DP:** Because $dp[i][j]$ depends on $dp[i + 1][j - 1]$, the outer loop for $i$ must run in reverse ($n - 1 \to 0$), ensuring $i + 1$ is precomputed before $i$.
3. **Contrast with Palindrome Partitioning II (LC 132):** LC 132 only asks for the *minimum number of cuts*, which can be solved purely in $\mathcal{O}(n^2)$ time using 1D DP on top of the palindrome table without any exponential backtracking.