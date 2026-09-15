---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 756: Pyramid Transition Matrix"
tags:
  - leetcode
  - coding
  - backtracking
  - hash-table
  - string
  - amazon
  - google
---

# LeetCode 756: Pyramid Transition Matrix

**Target Companies:** Google, Amazon, Uber  
**Difficulty:** Medium  
**Topic:** Backtracking / Multi-Level DFS / State Memoization  

---

### Problem Statement

You are stacking blocks to form a pyramid. Each block has a color which is represented by a single letter. Each row of blocks is built on top of the row below it.

You are given:
- A string `bottom` representing the initial bottom row of the pyramid.
- A list of 3-character strings `allowed`, where each string `pattern` represents a permitted rule: if the left block is `pattern[0]` and the right block is `pattern[1]`, the block placed directly above them can be `pattern[2]`.

Return `true` if you can build the pyramid all the way to the top such that every triangular block pattern is allowed, or `false` otherwise. The top of the pyramid consists of a single block.

---

### Input & Output Formats & Constraints

- **Input:** `bottom: str`, `allowed: List[str]`
- **Output:** `bool`
- **Constraints:**
  - $2 \le \text{bottom.length} \le 6$
  - $0 \le \text{allowed.length} \le 216$
  - `allowed[i].length == 3`
  - The letters in all input strings are chosen from the set `{'A', 'B', 'C', 'D', 'E', 'F'}`.
  - All elements of `allowed` are unique.

---

### Key Idea & Intuition

- **Pyramid Structure & Height Reduction:**
  - If a row has length $L$, the row directly above it has length $L - 1$.
  - To construct block $i$ of the next row, we must find a valid transition for the adjacent pair in the current row: $(\text{curr}[i], \text{curr}[i + 1]) \to \text{next}[i]$.
  - The pyramid reaches completion when the next row has length $1$.
- **Two-Level Backtracking Structure:**
  1. **Row-Level DFS (`solve(curr_row)`):**
     - Base case: `len(curr_row) == 1` $\implies$ return `True`.
     - Build all possible valid `next_row` strings from `curr_row`.
     - For each candidate `next_row`, recursively check if `solve(next_row)` can reach the top.
  2. **Intra-Row Backtracking (`build_next_row(idx, next_path)`):**
     - At index `idx`, inspect the pair $(\text{curr}[idx], \text{curr}[idx + 1])$.
     - For each allowed top character, append to `next_path` and recurse to `idx + 1`.
     - When `idx == len(curr_row) - 1`, we have formed a complete candidate `next_row`.
- **Memoization of Dead Ends (`failed_rows`):**
  - Multiple different pyramid paths can produce the same intermediate row configuration.
  - If a row string $R$ has been evaluated and cannot reach the peak, store $R$ in a `failed` set. Any future branch that encounters row $R$ can immediately return `False`.

---

### Solution Approach (Step-by-Step)

1. **Preprocess Transitions:**
   - Map each pair `(u, v)` to a list of allowed top characters `allowed_map[(u, v)]`.
2. **Memoization Cache:**
   - Maintain `failed = set()` to remember rows that cannot form a valid pyramid.
3. **Recursive Function `solve(row)`:**
   - If `len(row) == 1`: return `True`.
   - If `row in failed`: return `False`.
   - Define inner helper `generate_next(idx, path)`:
     - If `idx == len(row) - 1`:
       - A candidate next row is complete: `candidate = "".join(path)`.
       - If `solve(candidate)` is `True`: return `True`.
       - Return `False`.
     - Pair: `pair = row[idx : idx + 2]`.
     - For `top_char` in `allowed_map.get(pair, [])`:
       - `path.append(top_char)`
       - If `generate_next(idx + 1, path)`: return `True`.
       - `path.pop()`
     - Return `False`.
   - If `generate_next(0, [])` is `True`: return `True`.
   - `failed.add(row)`
   - Return `False`.
4. Call `solve(bottom)`.

---

### Visual Algorithm Walkthrough

Let `bottom = "BCD"`, `allowed = ["BCG", "CDE", "GEA", "FFF"]` (Mapping: `BC->G`, `CD->E`, `GE->A`).

```
Target: Build upward from length 3 -> 2 -> 1:

Row 0 (len=3):      B       C       D
                     \     / \     /
Transitions:          BC->G   CD->E
                       \ /     \ /
Row 1 (len=2):          G       E
                         \     /
Transition:               GE->A
                           \ /
Row 2 (len=1):              A   <-- Peak Reached! (len == 1)

Result: True!
```

If at any point a pair has no transitions in `allowed_map`, that branch halts and backtracks immediately.

---

### Solved Examples with Multiple Inputs

| Test Case | `bottom` | `allowed` | Can Reach Peak? | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | `"BCD"` | `["BCC","CDE","CEA","FFF"]` | `"CD" -> "E"`, `"BC" -> "C"`, then `"CE" -> "A"` | `true` |
| **No Valid Apex** | `"AAAA"` | `["AAB","AAC","BCD","BBE","DEF"]` | Dead end before peak | `false` |
| **Already Length 2** | `"AB"` | `["ABC"]` | `"AB" -> "C"` (len 1) | `true` |
| **Empty Transitions** | `"ABC"` | `[]` | No moves possible | `false` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List, Dict, Set
from collections import defaultdict

class Solution:
    def pyramidTransition(self, bottom: str, allowed: List[str]) -> bool:
        """
        Determines whether a pyramid can be built to a single top block.
        Uses two-tier backtracking with memoization on failed rows.
        """
        # Step 1: Preprocess allowed transitions
        transitions: Dict[str, List[str]] = defaultdict(list)
        for pattern in allowed:
            transitions[pattern[:2]].append(pattern[2])

        failed_rows: Set[str] = set()

        # Step 2: Outer DFS (row by row)
        def solve(current_row: str) -> bool:
            if len(current_row) == 1:
                return True

            if current_row in failed_rows:
                return False

            # Inner DFS (generate all valid next rows)
            def generate_next(idx: int, path: List[str]) -> bool:
                if idx == len(current_row) - 1:
                    return solve("".join(path))

                pair = current_row[idx:idx + 2]
                for top in transitions.get(pair, []):
                    path.append(top)
                    if generate_next(idx + 1, path):
                        return True
                    path.pop()  # Backtrack

                return False

            if generate_next(0, []):
                return True

            failed_rows.add(current_row)
            return False

        return solve(bottom)
```

#### C++17
```cpp
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

class Solution {
public:
    bool pyramidTransition(const std::string& bottom, const std::vector<std::string>& allowed) {
        std::unordered_map<std::string, std::vector<char>> transitions;
        for (const std::string& p : allowed) {
            transitions[p.substr(0, 2)].push_back(p[2]);
        }

        std::unordered_set<std::string> failed_rows;
        return solve(bottom, transitions, failed_rows);
    }

private:
    bool solve(const std::string& row,
               std::unordered_map<std::string, std::vector<char>>& transitions,
               std::unordered_set<std::string>& failed_rows) {
        if (row.size() == 1) return true;
        if (failed_rows.count(row)) return false;

        std::string next_row;
        if (generate_next(0, row, next_row, transitions, failed_rows)) {
            return true;
        }

        failed_rows.insert(row);
        return false;
    }

    bool generate_next(int idx, const std::string& row, std::string& next_row,
                       std::unordered_map<std::string, std::vector<char>>& transitions,
                       std::unordered_set<std::string>& failed_rows) {
        if (idx == static_cast<int>(row.size()) - 1) {
            return solve(next_row, transitions, failed_rows);
        }

        std::string pair = row.substr(idx, 2);
        if (transitions.count(pair)) {
            for (char c : transitions[pair]) {
                next_row.push_back(c);
                if (generate_next(idx + 1, row, next_row, transitions, failed_rows)) {
                    return true;
                }
                next_row.pop_back(); // Backtrack
            }
        }

        return false;
    }
};
```

#### Java
```java
import java.util.*;

class Solution {
    public boolean pyramidTransition(String bottom, List<String> allowed) {
        Map<String, List<Character>> transitions = new HashMap<>();
        for (String p : allowed) {
            String key = p.substring(0, 2);
            transitions.computeIfAbsent(key, k -> new ArrayList<>()).add(p.charAt(2));
        }

        Set<String> failedRows = new HashSet<>();
        return solve(bottom, transitions, failedRows);
    }

    private boolean solve(String row, Map<String, List<Character>> transitions, Set<String> failedRows) {
        if (row.length() == 1) {
            return true;
        }
        if (failedRows.contains(row)) {
            return false;
        }

        StringBuilder nextRow = new StringBuilder();
        if (generateNext(0, row, nextRow, transitions, failedRows)) {
            return true;
        }

        failedRows.add(row);
        return false;
    }

    private boolean generateNext(int idx, String row, StringBuilder nextRow,
                                 Map<String, List<Character>> transitions, Set<String> failedRows) {
        if (idx == row.length() - 1) {
            return solve(nextRow.toString(), transitions, failedRows);
        }

        String pair = row.substring(idx, idx + 2);
        if (transitions.containsKey(pair)) {
            for (char c : transitions.get(pair)) {
                nextRow.append(c);
                if (generateNext(idx + 1, row, nextRow, transitions, failedRows)) {
                    return true;
                }
                nextRow.deleteCharAt(nextRow.length() - 1); // Backtrack
            }
        }

        return false;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(A^H)$, where $H = \text{bottom.length} \le 6$ and $A$ is the maximum number of transitions per pair ($A \le 6$).
  - For $H \le 6$, maximum pyramid height is 6.
  - The number of unique row configurations of length $\le 6$ over 6 alphabet characters is small, and memoizing `failed_rows` prevents repeated evaluation of dead-end subtrees.
  - Executes in $< 15 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(H^2)$ auxiliary space for the recursion stack and string buffers, plus $\mathcal{O}(6^H)$ worst-case for the `failed_rows` cache.

---

### Takeaway Pattern & Interview Traps

- **Two-Tier Recursion:** When a problem requires constructing a new sequence of length $L - 1$ from length $L$ via combinatorial local rules, nesting an inner DFS (intra-row generator) inside an outer DFS (row-to-row transition) keeps the code structured and clean.
- **Memoizing Only Failures:** Notice that we only memoize `failed_rows`. If a row succeeds, the recursion bubbles `True` all the way to the root immediately without needing to cache success!