---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 1079: Letter Tile Possibilities"
tags:
  - leetcode
  - coding
  - backtracking
  - counting
  - amazon
  - google
---

# LeetCode 1079: Letter Tile Possibilities

**Target Companies:** Amazon, Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Backtracking / Frequency Map Permutations  

---

### Problem Statement

You have `n` tiles, where each tile has one letter `tiles[i]` printed on it.

Return *the number of possible non-empty sequences of letters* you can make using the letters printed on those `tiles`.

---

### Input & Output Formats & Constraints

- **Input:** `tiles: str` (a string of uppercase English letters)
- **Output:** `int` (total count of unique, non-empty sequences)
- **Constraints:**
  - $1 \le \text{tiles.length} \le 7$
  - `tiles` consists of uppercase English letters.

---

### Key Idea & Intuition

- **Counting Every Prefix / Node, Not Just Leaves:**
  - Standard permutations count sequences of full length $N$. Here, sequences of **any length** from $1$ to $N$ are valid.
  - In a state-space recursion tree, every non-empty state visited represents a unique, valid word! Thus, every valid recursive call increments the count by $1$.
- **Handling Duplicates via Frequency Counting:**
  - If we sort and use a `used` boolean array, we must carefully prune duplicates at each level (`if i > 0 and tiles[i] == tiles[i-1] and not used[i-1]: continue`).
  - An even cleaner and more optimal approach is using a **frequency array** of size 26 (or hash map).
  - At each recursion level, iterate over all 26 uppercase letters:
    - If `count[c] > 0`, we pick character `c`, decrement `count[c]`, add 1 to our total count, and recursively explore subsequent characters.
    - Upon returning, backtrack by incrementing `count[c]`.
  - Because we branch only on distinct available characters at each step, **no duplicate sequence is ever generated**, eliminating any need for a `Set` or complex duplicate skip conditions.

---

### Solution Approach (Step-by-Step)

1. **Build Frequency Array:**
   - Initialize an array `counts` of size 26 with counts of each character in `tiles`.
2. **Recursive Backtracking Function `dfs()`:**
   - Initialize `total = 0`.
   - Loop `i` from $0$ to $25$:
     - If `counts[i] == 0`, continue.
     - We can form a new unique prefix by appending character `chr(ord('A') + i)`:
       - `total += 1`
       - `counts[i] -= 1`
       - `total += dfs()`
       - `counts[i] += 1` (backtrack)
   - Return `total`.
3. **Return Value:**
   - The result of calling `dfs()`.

---

### Visual Algorithm Walkthrough

Let `tiles = "AAB"`. Character counts: `{'A': 2, 'B': 1}`.

```
                    Root (counts: A:2, B:1)
                   /                       \
          Pick 'A'                           Pick 'B'
       (counts: A:1, B:1)                (counts: A:2, B:0)
         total += 1 [seq="A"]              total += 1 [seq="B"]
        /                  \                     |
   Pick 'A'              Pick 'B'              Pick 'A'
 (counts: A:0, B:1)    (counts: A:1, B:0)   (counts: A:1, B:0)
  total += 1            total += 1           total += 1
  [seq="AA"]            [seq="AB"]           [seq="BA"]
      |                     |                    |
   Pick 'B'              Pick 'A'             Pick 'A'
 (counts: A:0, B:0)    (counts: A:0, B:0)   (counts: A:0, B:0)
  total += 1            total += 1           total += 1
  [seq="AAB"]           [seq="ABA"]          [seq="BAA"]

Total unique non-empty sequences = 8:
- Length 1: "A", "B" (2)
- Length 2: "AA", "AB", "BA" (3)
- Length 3: "AAB", "ABA", "BAA" (3)
Total = 2 + 3 + 3 = 8.
```

---

### Solved Examples with Multiple Inputs

| Test Case | `tiles` | Unique Frequencies | Valid Sequences | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Example 1** | `"AAB"` | `A: 2, B: 1` | `A`, `B`, `AA`, `AB`, `BA`, `AAB`, `ABA`, `BAA` | `8` |
| **Example 2** | `"AAABBC"` | `A: 3, B: 2, C: 1` | All non-empty unique permutations | `188` |
| **Single Char** | `"V"` | `V: 1` | `V` | `1` |
| **All Identical** | `"AAAA"` | `A: 4` | `A`, `AA`, `AAA`, `AAAA` | `4` |
| **All Distinct** | `"ABC"` | `A: 1, B: 1, C: 1` | $3 + 6 + 6 = 15$ | `15` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def numTilePossibilities(self, tiles: str) -> int:
        """
        Calculates the number of possible non-empty sequences of letters.
        Uses frequency count backtracking to avoid generating duplicate branches.
        """
        counts = [0] * 26
        for ch in tiles:
            counts[ord(ch) - ord('A')] += 1

        def dfs() -> int:
            total = 0
            for i in range(26):
                if counts[i] == 0:
                    continue
                # Pick character i: represents 1 new valid sequence
                total += 1
                counts[i] -= 1
                # Recurse for remaining positions
                total += dfs()
                # Backtrack
                counts[i] += 1
            return total

        return dfs()
```

#### C++17
```cpp
#include <string>
#include <vector>
#include <array>

class Solution {
public:
    int numTilePossibilities(const std::string& tiles) {
        std::array<int, 26> counts{};
        for (char c : tiles) {
            counts[c - 'A']++;
        }
        return dfs(counts);
    }

private:
    int dfs(std::array<int, 26>& counts) {
        int total = 0;
        for (int i = 0; i < 26; ++i) {
            if (counts[i] == 0) continue;
            
            // Choose character i
            total += 1;
            counts[i]--;
            
            // Explore sub-sequences
            total += dfs(counts);
            
            // Backtrack
            counts[i]++;
        }
        return total;
    }
};
```

#### Java
```java
class Solution {
    public int numTilePossibilities(String tiles) {
        int[] counts = new int[26];
        for (char c : tiles.toCharArray()) {
            counts[c - 'A']++;
        }
        return dfs(counts);
    }

    private int dfs(int[] counts) {
        int total = 0;
        for (int i = 0; i < 26; i++) {
            if (counts[i] == 0) continue;

            // Form 1 new sequence with counts[i]
            total++;
            counts[i]--;

            // Recursively count following sequences
            total += dfs(counts);

            // Backtrack
            counts[i]++;
        }
        return total;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(26 \times \text{Number of Unique Sequences})$.
  - For $N \le 7$, the maximum number of unique sequences occurs when all characters are distinct: $\sum_{k=1}^7 \frac{7!}{(7-k)!} = 7 + 42 + 210 + 840 + 2520 + 5040 + 5040 = 13,699$.
  - At each state, we iterate across 26 alphabet characters. Total operations $\approx 26 \times 13,699 \approx 3.5 \times 10^5$, running in $< 5 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(N)$ where $N = \text{tiles.length} \le 7$.
  - Recursion depth is bounded by the length of the string ($N \le 7$).
  - Auxiliary frequency array takes $\mathcal{O}(1)$ space ($\Sigma = 26$).

---

### Takeaway Pattern & Interview Traps

- **Node-Counting vs Leaf-Counting:** Unlike standard permutation problems where you only count or collect when `len(path) == n`, here every step in the DFS tree represents a non-empty sequence. Increment the count as soon as a node is chosen.
- **Frequency Map avoids Sorting & `used` arrays:** Branching over unique character counts automatically handles duplicates without needing to sort the input or check `if i > 0 and tiles[i] == tiles[i-1] and not used[i-1]`.
- **String Construction Overhead:** Notice we never build or store the actual strings! Since only the count is requested, maintaining character counts directly avoids high string concatenation and hash set overhead.