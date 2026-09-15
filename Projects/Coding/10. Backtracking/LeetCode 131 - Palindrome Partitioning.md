---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 131: Palindrome Partitioning"
tags:
  - leetcode
  - coding
  - backtracking
  - dynamic-programming
  - string
  - amazon
  - google
---

# LeetCode 131: Palindrome Partitioning

**Target Companies:** Amazon, Google, Meta, Bloomberg, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Backtracking / Dynamic Programming Precomputation / String Partitioning  

---

### Problem Statement

Given a string `s`, partition `s` such that every substring of the partition is a **palindrome**. Return *all possible palindrome partitionings of* `s`.

A **palindrome** string is a string that reads the same backward as forward.

---

### Input & Output Formats & Constraints

- **Input:** `s: str`
- **Output:** `List[List[str]]` containing all partition combinations.
- **Constraints:**
  - $1 \le \text{s.length} \le 16$
  - `s` contains only lowercase English letters.

---

### Key Idea & Intuition

- **String Partitioning as Decision Tree:**
  - A string of length $N$ has $N - 1$ potential split positions (between adjacent characters). Each split can either be chosen or skipped, yielding up to $2^{N-1}$ total possible partitions.
  - At each index `start`, we iterate through all possible next cut points `end` from `start` to $N - 1$.
  - If substring `s[start...end]` is a palindrome, we make the cut, add `s[start...end]` to our `current_partition`, and recursively partition the suffix starting at `end + 1`.
  - If `start == N`, we have reached the end of the string with valid palindromic parts, so we record a copy of `current_partition`.
- **Optimization via DP Palindrome Table:**
  - Checking if a substring `s[i...j]` is a palindrome with two pointers takes $\mathcal{O}(N)$. Doing this at every branch adds an $\mathcal{O}(N)$ factor.
  - Instead, we can precompute a 2D boolean table `is_palindrome[i][j]` using DP:
    $$\text{is\_palindrome}[i][j] = (s[i] == s[j]) \land (j - i \le 2 \lor \text{is\_palindrome}[i+1][j-1])$$
  - Precomputing takes $\mathcal{O}(N^2)$, reducing every palindrome check during backtracking to an instantaneous $\mathcal{O}(1)$ lookup.

---

### Solution Approach (Step-by-Step)

1. **Precompute Palindrome DP Table:**
   - Create an $N \times N$ boolean table `dp`.
   - Populate diagonally or length-by-length (or bottom-up from $i = N - 1$ down to $0$):
     - `dp[i][j] = (s[i] == s[j]) and (j - i <= 2 or dp[i + 1][j - 1])`
2. **Backtrack `dfs(start)`:**
   - **Base Case:** If `start == len(s)`, append a copy of `current_partition` to `results` and return.
   - **Branching:** For `end` from `start` to $N - 1$:
     - If `dp[start][end]` is `True`:
       - `current_partition.append(s[start : end + 1])`
       - `dfs(end + 1)`
       - `current_partition.pop()` (backtrack)
3. Return `results`.

---

### Visual Algorithm Walkthrough

Let `s = "aab"`. Precomputed palindromes: `(0,0): "a"`, `(0,1): "aa"`, `(1,1): "a"`, `(2,2): "b"`.

```
                       dfs(start = 0, path = [])
                      /                         \
           cut at end=0: "a"                 cut at end=1: "aa"
           dfs(1, ["a"])                      dfs(2, ["aa"])
          /             \                            |
   cut end=1: "a"     cut end=2: "ab"           cut end=2: "b"
   dfs(2, ["a","a"])  (NOT PALINDROME - PRUNED)  dfs(3, ["aa","b"])
         |                                           |
   cut end=2: "b"                               Base Case: start == 3
   dfs(3, ["a","a","b"])                        Add ["aa", "b"]
         |
   Base Case: start == 3
   Add ["a", "a", "b"]

Output: [["a", "a", "b"], ["aa", "b"]]
```

---

### Solved Examples with Multiple Inputs

| Test Case | `s` | Substring Palindromes | Valid Partitions | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | `"aab"` | `"a"`, `"aa"`, `"a"`, `"b"` | `["a","a","b"]`, `["aa","b"]` | `[["a","a","b"],["aa","b"]]` |
| **Single Character** | `"a"` | `"a"` | `["a"]` | `[["a"]]` |
| **All Identical** | `"aaa"` | `"a"`, `"aa"`, `"aaa"` | `["a","a","a"]`, `["a","aa"]`, `["aa","a"]`, `["aaa"]` | `[["a","a","a"],["a","aa"],["aa","a"],["aaa"]]` |
| **All Different** | `"abc"` | `"a"`, `"b"`, `"c"` | Only single character splits | `[["a","b","c"]]` |
| **Even Palindrome** | `"abba"` | `"a"`, `"b"`, `"bb"`, `"abba"` | `["a","b","b","a"]`, `["a","bb","a"]`, `["abba"]` | `[["a","b","b","a"],["a","bb","a"],["abba"]]` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def partition(self, s: str) -> List[List[str]]:
        """
        Partitions string s into all possible lists of palindromic substrings.
        Uses DP precomputation for O(1) palindrome checks during DFS backtracking.
        """
        n = len(s)
        # dp[i][j] is True if s[i..j] is a palindrome
        dp = [[False] * n for _ in range(n)]
        
        # Bottom-up DP to compute all palindromic substrings
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i <= 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True

        results: List[List[str]] = []
        current_partition: List[str] = []

        def backtrack(start: int) -> None:
            if start == n:
                results.append(list(current_partition))
                return

            for end in range(start, n):
                if dp[start][end]:
                    # Choose
                    current_partition.append(s[start : end + 1])
                    # Explore
                    backtrack(end + 1)
                    # Backtrack
                    current_partition.pop()

        backtrack(0)
        return results
```

#### C++17
```cpp
#include <string>
#include <vector>

class Solution {
public:
    std::vector<std::vector<std::string>> partition(const std::string& s) {
        int n = static_cast<int>(s.size());
        // dp[i][j] stores whether s[i..j] is a palindrome
        std::vector<std::vector<bool>> dp(n, std::vector<bool>(n, false));

        for (int i = n - 1; i >= 0; --i) {
            for (int j = i; j < n; ++j) {
                if (s[i] == s[j] && (j - i <= 2 || dp[i + 1][j - 1])) {
                    dp[i][j] = true;
                }
            }
        }

        std::vector<std::vector<std::string>> results;
        std::vector<std::string> current_partition;
        backtrack(0, s, dp, current_partition, results);
        return results;
    }

private:
    void backtrack(int start, const std::string& s, const std::vector<std::vector<bool>>& dp,
                   std::vector<std::string>& current_partition,
                   std::vector<std::vector<std::string>>& results) {
        if (start == static_cast<int>(s.size())) {
            results.push_back(current_partition);
            return;
        }

        for (int end = start; end < static_cast<int>(s.size()); ++end) {
            if (dp[start][end]) {
                // Choose
                current_partition.push_back(s.substr(start, end - start + 1));
                // Explore
                backtrack(end + 1, s, dp, current_partition, results);
                // Backtrack
                current_partition.pop_back();
            }
        }
    }
};
```

#### Java
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<List<String>> partition(String s) {
        int n = s.length();
        boolean[][] dp = new boolean[n][n];

        // Precompute palindrome lookup table
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (s.charAt(i) == s.charAt(j) && (j - i <= 2 || dp[i + 1][j - 1])) {
                    dp[i][j] = true;
                }
            }
        }

        List<List<String>> results = new ArrayList<>();
        List<String> currentPartition = new ArrayList<>();
        backtrack(0, s, dp, currentPartition, results);
        return results;
    }

    private void backtrack(int start, String s, boolean[][] dp, 
                           List<String> currentPartition, 
                           List<List<String>> results) {
        if (start == s.length()) {
            results.add(new ArrayList<>(currentPartition));
            return;
        }

        for (int end = start; end < s.length(); end++) {
            if (dp[start][end]) {
                // Choose
                currentPartition.add(s.substring(start, end + 1));
                // Explore
                backtrack(end + 1, s, dp, currentPartition, results);
                // Backtrack
                currentPartition.remove(currentPartition.size() - 1);
            }
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \cdot 2^N)$.
  - DP precomputation takes $\mathcal{O}(N^2)$ time.
  - In the worst case (where all characters are identical, e.g. `"aaaa"`), every substring is a palindrome, resulting in $2^{N-1}$ total partitions. For each partition, constructing and copying substrings of total length $N$ takes $\mathcal{O}(N)$.
  - Overall time is $\mathcal{O}(N^2 + N \cdot 2^N) = \mathcal{O}(N \cdot 2^N)$. Since $N \le 16$, $16 \cdot 2^{16} \approx 10^6$ operations, executing in under $10 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(N^2)$ auxiliary space.
  - The DP table requires $\mathcal{O}(N^2)$ space.
  - The recursion stack takes $\mathcal{O}(N)$ depth, and `current_partition` uses at most $\mathcal{O}(N)$ elements.

---

### Takeaway Pattern & Interview Traps

- **Partitioning vs Subsets:** In subset/combination problems, elements can be picked out of order. In partitioning problems, the chosen parts must cover the string contiguously from index $0$ to $N - 1$.
- **DP Precomputation vs Inline Check:** Without DP precomputation, checking each substring takes $\mathcal{O}(N)$ at each recursive branch, yielding $\mathcal{O}(N^2 \cdot 2^N)$ time. Precomputing `dp[i][j]` reduces every check to $\mathcal{O}(1)$.
- **Substring Copy Overhead:** In languages with string copies (like Java `substring` or C++ `s.substr`), string slicing creates new objects. Passing `(start, end)` indices and taking substrings only during insertion minimizes string allocation.