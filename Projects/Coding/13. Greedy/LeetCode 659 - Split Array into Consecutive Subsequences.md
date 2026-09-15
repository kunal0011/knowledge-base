---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 659: Split Array into Consecutive Subsequences"
tags:
  - leetcode
  - coding
  - greedy
  - hash-table
  - array
  - amazon
  - google
---

# LeetCode 659: Split Array into Consecutive Subsequences

**Target Companies:** Google, Amazon, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Hash Table / Counting  

---

### Problem Statement

You are given an integer array `nums` that is **sorted in non-decreasing order**.

Determine if it is possible to split `nums` into **one or more subsequences** such that:
1. Each subsequence is a **consecutive increasing sequence** (i.e. each element is exactly 1 greater than the previous element).
2. Each subsequence has a length of **at least 3**.

Return `true` if you can make this split, or `false` otherwise.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` — sorted non-decreasing array ($1 \le \text{nums.length} \le 10^4$).
- **Output:**
  - `bool` — `true` if valid decomposition into consecutive subsequences of length $\ge 3$ is possible, else `false`.
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^4$
  - $-1000 \le \text{nums}[i] \le 1000$
  - `nums` is sorted in non-decreasing order.

---

### Key Idea & Intuition

When processing each number $x$ in sorted order, we need to decide what to do with $x$.
There are two possibilities:
1. **Append to an existing subsequence:** Attach $x$ to an already existing subsequence that ended at $x - 1$.
2. **Start a new 3-element subsequence:** Group $x$ with $x + 1$ and $x + 2$ to seed a new valid subsequence.

#### The Greedy Priority Choice:
Should we prefer Option 1 or Option 2?
> **Always prefer appending to an existing subsequence (Option 1) first!**

**Why?**
- Existing subsequences are "hungry" to stay valid or be extended.
- Seeding a new subsequence requires consuming two future numbers ($x + 1$ and $x + 2$).
- If we started a new sequence with $x$ instead of attaching it to the sequence ending at $x - 1$, that previous sequence would be left permanently stranded at length $< 3$ or unable to extend, and $x + 1, x + 2$ would be prematurely spent.
- Extending an existing subsequence only consumes $x$ itself, preserving maximum flexibility for future numbers.

#### Data Structures:
- `count[x]`: remaining available count of number $x$.
- `end[x]`: count of valid consecutive subsequences that currently terminate at number $x$.

---

### Solution Approach (Step-by-Step)

1. Precompute `count = Counter(nums)` and initialize an empty hash map `end = defaultdict(int)`.
2. Iterate through each number $x$ in `nums`:
   - If `count[x] == 0`:
     - $x$ was already used in a 3-element seed $\rightarrow$ continue.
   - **Option 1 (Append to existing):**
     - If `end[x - 1] > 0`:
       - `end[x - 1] -= 1`
       - `end[x] += 1`
       - `count[x] -= 1`
   - **Option 2 (Create new triplet `[x, x+1, x+2]`):**
     - Else if `count[x + 1] > 0` and `count[x + 2] > 0`:
       - `count[x] -= 1`
       - `count[x + 1] -= 1`
       - `count[x + 2] -= 1`
       - `end[x + 2] += 1`
   - **Option 3 (Failure):**
     - Else: $x$ cannot be placed anywhere $\rightarrow$ return `False`.
3. If all numbers are successfully placed, return `True`.

---

### Visual Algorithm Walkthrough

For `nums = [1, 2, 3, 3, 4, 5]`:
Initial `count = {1:1, 2:1, 3:2, 4:1, 5:1}`
`end = {}`

```
Process x = 1:
  end[0] == 0. Can start triplet? count[2]>0, count[3]>0 -> YES!
  Create [1, 2, 3]:
  count: {1:0, 2:0, 3:1, 4:1, 5:1}
  end[3] = 1

Process x = 2:
  count[2] == 0 -> SKIP

Process x = 3 (first occurrence):
  count[3] == 1 (one was used in [1,2,3])
  Can append to end[2]? end[2] == 0.
  Can start triplet? count[4]>0, count[5]>0 -> YES!
  Create [3, 4, 5]:
  count: {3:0, 4:0, 5:0}
  end[5] = 1

Process x = 3 (second occurrence): count[3] == 0 -> SKIP
Process x = 4: count[4] == 0 -> SKIP
Process x = 5: count[5] == 0 -> SKIP

Loop completed successfully.
Return True.
Subsequences: [1, 2, 3] and [3, 4, 5].
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [1, 2, 3, 3, 4, 5]`
- **Output:** `true`

#### Example 2:
- **Input:** `nums = [1, 2, 3, 3, 4, 4, 5, 5]`
- **Tracing:** Split into `[1, 2, 3, 4, 5]` and `[3, 4, 5]`.
- **Output:** `true`

#### Example 3 (Impossible):
- **Input:** `nums = [1, 2, 3, 4, 4, 5]`
- **Tracing:**
  - Seed `[1, 2, 3]` (end at 3)
  - Append 4 $\rightarrow$ `[1, 2, 3, 4]` (end at 4)
  - Next 4: cannot append to 3. Needs 5, 6, but only 5 exists. Cannot form length 3!
- **Output:** `false`

---

### Multi-Language Implementations

#### Python 3
```python
from collections import Counter, defaultdict
from typing import List

class Solution:
    def isPossible(self, nums: List[int]) -> bool:
        count = Counter(nums)
        end = defaultdict(int)
        
        for x in nums:
            if count[x] == 0:
                continue
                
            # Priority 1: Append to an existing subsequence ending at x - 1
            if end[x - 1] > 0:
                end[x - 1] -= 1
                end[x] += 1
                count[x] -= 1
            # Priority 2: Create a new 3-element subsequence [x, x + 1, x + 2]
            elif count[x + 1] > 0 and count[x + 2] > 0:
                count[x] -= 1
                count[x + 1] -= 1
                count[x + 2] -= 1
                end[x + 2] += 1
            else:
                return False
                
        return True
```

#### C++17
```cpp
#include <vector>
#include <unordered_map>

class Solution {
public:
    bool isPossible(const std::vector<int>& nums) {
        std::unordered_map<int, int> count;
        std::unordered_map<int, int> end;
        
        for (int x : nums) {
            count[x]++;
        }
        
        for (int x : nums) {
            if (count[x] == 0) continue;
            
            // Priority 1: Append to existing subsequence ending at x - 1
            if (end[x - 1] > 0) {
                end[x - 1]--;
                end[x]++;
                count[x]--;
            }
            // Priority 2: Seed new subsequence [x, x + 1, x + 2]
            else if (count[x + 1] > 0 && count[x + 2] > 0) {
                count[x]--;
                count[x + 1]--;
                count[x + 2]--;
                end[x + 2]++;
            }
            else {
                return false;
            }
        }
        
        return true;
    }
};
```

#### Java 17
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public boolean isPossible(int[] nums) {
        Map<Integer, Integer> count = new HashMap<>();
        Map<Integer, Integer> end = new HashMap<>();
        
        for (int x : nums) {
            count.put(x, count.getOrDefault(x, 0) + 1);
        }
        
        for (int x : nums) {
            if (count.get(x) == 0) {
                continue;
            }
            
            // Priority 1: Append to subsequence ending at x - 1
            if (end.getOrDefault(x - 1, 0) > 0) {
                end.put(x - 1, end.get(x - 1) - 1);
                end.put(x, end.getOrDefault(x, 0) + 1);
                count.put(x, count.get(x) - 1);
            }
            // Priority 2: Seed new subsequence [x, x + 1, x + 2]
            else if (count.getOrDefault(x + 1, 0) > 0 && count.getOrDefault(x + 2, 0) > 0) {
                count.put(x, count.get(x) - 1);
                count.put(x + 1, count.get(x + 1) - 1);
                count.put(x + 2, count.get(x + 2) - 1);
                end.put(x + 2, end.getOrDefault(x + 2, 0) + 1);
            }
            else {
                return false;
            }
        }
        
        return true;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - Building `count` hash map takes $\mathcal{O}(n)$ time.
  - Traversing `nums` performs $\mathcal{O}(1)$ average hash map queries and updates per element.
  - Total time is $\mathcal{O}(n)$.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space
  - The `count` and `end` hash maps store at most $n$ distinct integer keys.

---

### Takeaway Pattern & Interview Traps

- **Subsequence Extension vs Initialization:** In consecutive sequence packing, extending an established sequence must strictly take precedence over initializing a new sequence to satisfy the minimum length constraint without wasting future elements.
- **Pre-Decrementing Look-ahead Counters:** When seeding a new triplet `[x, x + 1, x + 2]`, make sure to decrement `count[x + 1]` and `count[x + 2]` immediately so future iterations accurately reflect remaining quantities.