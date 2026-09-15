---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 128: Longest Consecutive Sequence"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - hash-table
  - union-find
  - amazon
  - google
  - meta
---

# LeetCode 128: Longest Consecutive Sequence

**Target Companies:** Amazon, Google (Top Classic), Meta, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Hash Set / Sequence Start Detection

---

### Problem Statement

Given an unsorted array of integers `nums`, return *the length of the longest consecutive elements sequence*.

You must write an algorithm that runs in **$\mathcal{O}(n)$ time**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `int` (maximum consecutive sequence length)
- **Constraints:**
  - $0 \le \text{nums.length} \le 10^5$
  - $-10^9 \le \text{nums}[i] \le 10^9$

---

### Key Idea & Intuition

#### 1. Why Not Sort?
Sorting requires $\mathcal{O}(n \log n)$ time, which violates the strict $\mathcal{O}(n)$ requirement.

#### 2. Sequence Head Invariant:
Insert all numbers into a hash set for $\mathcal{O}(1)$ average lookups:
- A number $x$ is the **start of a consecutive streak** if and only if $x - 1 \notin \text{set}$.
- If $x - 1 \in \text{set}$, then $x$ is merely an internal node of a longer streak starting somewhere at or before $x - 1$. If we initiated a streak count from $x$, we would repeat work and risk quadratic $\mathcal{O}(n^2)$ time.
- By checking `if num - 1 not in set:`, we strictly only count forward from sequence heads!
- Because each element belongs to exactly one connected component, each number is traversed in the `while` loop at most once. Hence, the overall time is amortized $\mathcal{O}(n)$.

---

### Solution Approach (Step-by-Step)

1. **Edge Case:**
   - If `nums` is empty, return 0.
2. **Populate Hash Set:**
   - Create a set `num_set = set(nums)` to remove duplicates and enable $\mathcal{O}(1)$ lookup.
3. **Scan for Sequence Roots:**
   - Initialize `max_streak = 0`.
   - Iterate over each `num` in `num_set`:
     - If `num - 1` is **not** in `num_set`:
       - This is a new sequence root.
       - Let `curr_num = num` and `curr_streak = 1`.
       - While `curr_num + 1 in num_set`:
         - `curr_num += 1`
         - `curr_streak += 1`
       - Update `max_streak = max(max_streak, curr_streak)`.
4. **Return Result:**
   - Return `max_streak`.

---

### Visual Algorithm Walkthrough

#### Example: `nums = [100, 4, 200, 1, 3, 2]`
`num_set = {1, 2, 3, 4, 100, 200}`

```
Examine 100:
  100 - 1 = 99 in set? NO -> Root found!
  Check 101: NO
  Streak = 1. max_streak = 1.

Examine 4:
  4 - 1 = 3 in set? YES -> Skip (not a root).

Examine 200:
  200 - 1 = 199 in set? NO -> Root found!
  Check 201: NO
  Streak = 1. max_streak = 1.

Examine 1:
  1 - 1 = 0 in set? NO -> Root found!
  Check 2: YES (streak = 2)
  Check 3: YES (streak = 3)
  Check 4: YES (streak = 4)
  Check 5: NO
  max_streak = max(1, 4) = 4.

Examine 3:
  3 - 1 = 2 in set? YES -> Skip.

Examine 2:
  2 - 1 = 1 in set? YES -> Skip.

Final max_streak = 4 (Sequence: [1, 2, 3, 4]).
```

---

### Solved Examples with Multiple Inputs

| Input `nums` | Unique Elements in Set | Roots Tested | Max Consecutive Sequence | Output |
| :--- | :--- | :--- | :--- | :--- |
| `[100, 4, 200, 1, 3, 2]` | `{1, 2, 3, 4, 100, 200}` | `100` (len 1), `200` (len 1), `1` (len 4) | `[1, 2, 3, 4]` | `4` |
| `[0,3,7,2,5,8,4,6,0,1]` | `{0, 1, 2, 3, 4, 5, 6, 7, 8}` | `0` (len 9) | `[0, 1, 2, 3, 4, 5, 6, 7, 8]` | `9` |
| `[]` | `{}` | None | None | `0` |
| `[1]` | `{1}` | `1` (len 1) | `[1]` | `1` |
| `[9, 1, 4, 7, 3, -1, 0, 5, 8, -1, 6]` | `{-1, 0, 1, 3, 4, 5, 6, 7, 8, 9}` | `-1` (len 3), `3` (len 7) | `[3, 4, 5, 6, 7, 8, 9]` | `7` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        max_streak = 0
        
        for num in num_set:
            # Only expand if num is the start of a sequence
            if num - 1 not in num_set:
                curr_num = num
                curr_streak = 1
                
                while curr_num + 1 in num_set:
                    curr_num += 1
                    curr_streak += 1
                    
                max_streak = max(max_streak, curr_streak)
                
        return max_streak
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_set>
#include <algorithm>

class Solution {
public:
    int longestConsecutive(std::vector<int>& nums) {
        std::unordered_set<int> numSet(nums.begin(), nums.end());
        int maxStreak = 0;

        for (int num : numSet) {
            // Only begin counting if num is the root of a streak
            if (!numSet.count(num - 1)) {
                int currNum = num;
                int currStreak = 1;

                while (numSet.count(currNum + 1)) {
                    currNum++;
                    currStreak++;
                }

                maxStreak = std::max(maxStreak, currStreak);
            }
        }
        return maxStreak;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.HashSet;
import java.util.Set;

class Solution {
    public int longestConsecutive(int[] nums) {
        Set<Integer> numSet = new HashSet<>();
        for (int num : nums) {
            numSet.add(num);
        }

        int maxStreak = 0;

        for (int num : numSet) {
            // Only begin counting if num is the root of a streak
            if (!numSet.contains(num - 1)) {
                int currNum = num;
                int currStreak = 1;

                while (numSet.contains(currNum + 1)) {
                    currNum++;
                    currStreak++;
                }

                maxStreak = Math.max(maxStreak, currStreak);
            }
        }
        return maxStreak;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$. Inserting all elements into the hash set takes $\mathcal{O}(n)$ time. In the search loop, each number is checked as a root in $\mathcal{O}(1)$. The inner `while` loop only triggers for the smallest number of each sequence, visiting each number at most once across the entire runtime. Hence, total time is strictly linear $\mathcal{O}(n)$.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space to store the elements in the hash set.

---

### Takeaway Pattern & Interview Traps

1. **The Sequence Root Guard:**
   - Without `if (num - 1 not in num_set)`, iterating forward for every element would take $\mathcal{O}(n^2)$ time on sorted arrays like `[1, 2, 3, ..., n]`. The single `num - 1 not in num_set` check is the crucial trick that guarantees linear time.
2. **Iterating Over the Set vs Array:**
   - Always iterate `for num in num_set` instead of `for num in nums`. If the input has many duplicate elements, iterating over the set avoids redundant work.
