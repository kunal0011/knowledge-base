---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 219: Contains Duplicate II"
tags:
  - leetcode
  - coding
  - sliding-window
  - array
  - hash-table
  - amazon
  - google
---

# LeetCode 219: Contains Duplicate II

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple  
**Difficulty:** Easy  
**Topic:** Sliding Window / Hash Set / Distance Invariant  

---

### Problem Statement

Given an integer array `nums` and an integer `k`, return `true` *if there are two distinct indices `i` and `j` in the array such that `nums[i] == nums[j]` and `abs(i - j) <= k`*. Otherwise, return `false`.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `k: int`
- **Output:** `bool`
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $-10^9 \le \text{nums}[i] \le 10^9$
  - $0 \le k \le 10^5$

---

### Key Idea & Intuition

- **The Sliding Window Set ($W \le k$):**
  - We only care about duplicates within a window of length at most $k$.
  - Maintain a hash set `window` that stores the elements present in the last $k$ indices: $[i - k, i - 1]$.
  - For each element `nums[i]`:
    1. If `nums[i]` is already in `window`, we found two identical numbers within distance $k \implies$ return `true`.
    2. Add `nums[i]` to `window`.
    3. If `len(window) > k`, remove the oldest element `nums[i - k]` to maintain maximum window size $k$.
  - This bounds the auxiliary space strictly to $\mathcal{O}(\min(N, k))$!

- **Alternative: Last-Seen Index Map:**
  - Map each number to its most recent index: `last_seen[val] = idx`.
  - If `val in last_seen and i - last_seen[val] <= k`: return `true`.
  - Update `last_seen[val] = i`.

---

### Solution Approach (Step-by-Step)

#### Sliding Window Set Approach:
1. Initialize an empty hash set `window = set()`.
2. Loop `i` from $0$ to $\text{len}(nums) - 1$:
   - If `i > k`:
     - Remove `nums[i - k - 1]` from `window`.
   - If `nums[i] in window`:
     - Return `True`.
   - Add `nums[i]` to `window`.
3. If loop finishes without finding duplicates, return `False`.

---

### Visual Algorithm Walkthrough

Let `nums = [1, 2, 3, 1]`, $k = 3$.

```
Indices:   0  1  2  3
Nums:     [1, 2, 3, 1]

i=0, num=1:
- window is empty
- Add 1 -> window = {1}

i=1, num=2:
- 2 not in {1}
- Add 2 -> window = {1, 2}

i=2, num=3:
- 3 not in {1, 2}
- Add 3 -> window = {1, 2, 3}

i=3, num=1:
- 1 IS IN window {1, 2, 3}!
- Distance between index 3 and earlier 1 (index 0) is 3 - 0 = 3 <= k (3).
- Return True immediately!
```

---

### Solved Examples with Multiple Inputs

| Test Case | `nums` | `k` | Duplicate Indices | Distance | Output |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Example 1** | `[1, 2, 3, 1]` | `3` | `i=0, j=3` | $|3 - 0| = 3 \le 3$ | `true` |
| **Example 2** | `[1, 0, 1, 1]` | `1` | `i=2, j=3` | $|3 - 2| = 1 \le 1$ | `true` |
| **Example 3** | `[1, 2, 3, 1, 2, 3]` | `2` | `i=0, j=3` | $|3 - 0| = 3 > 2$ | `false` |
| **k == 0** | `[1, 2, 1]` | `0` | $i$ and $j$ must be distinct ($|i - j| \ge 1$) | $|i - j| > 0$ always | `false` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        """
        Determines if there exist duplicate elements within distance k.
        Uses a sliding window set of capacity at most k for O(min(N, k)) space.
        """
        window = set()

        for i, val in enumerate(nums):
            # Keep window size at most k
            if i > k:
                window.remove(nums[i - k - 1])

            if val in window:
                return True

            window.add(val)

        return False
```

#### C++17
```cpp
#include <vector>
#include <unordered_set>

class Solution {
public:
    bool containsNearbyDuplicate(const std::vector<int>& nums, int k) {
        std::unordered_set<int> window;

        for (int i = 0; i < static_cast<int>(nums.size()); ++i) {
            if (i > k) {
                window.erase(nums[i - k - 1]);
            }

            if (window.count(nums[i])) {
                return true;
            }

            window.insert(nums[i]);
        }

        return false;
    }
};
```

#### Java
```java
import java.util.HashSet;
import java.util.Set;

class Solution {
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Set<Integer> window = new HashSet<>();

        for (int i = 0; i < nums.length; i++) {
            if (i > k) {
                window.remove(nums[i - k - 1]);
            }

            if (!window.add(nums[i])) {
                return true; // add() returns false if element was already present
            }
        }

        return false;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ where $N = \text{nums.length}$.
  - We make a single linear pass over the array.
  - Set insertion, lookup, and deletion take $\mathcal{O}(1)$ average time.
  - Overall time is $\mathcal{O}(N)$, completing in $< 15 \text{ ms}$ for $N = 10^5$.
- **Space Complexity:** $\mathcal{O}(\min(N, k))$ auxiliary space.
  - The hash set contains at most $k + 1$ elements at any moment.

---

### Takeaway Pattern & Interview Traps

- **Space Optimization via Sliding Set:** While a hash map storing all indices takes $\mathcal{O}(N)$ space, restricting the set to size $k$ guarantees $\mathcal{O}(\min(N, k))$ memory, which is significantly more space-efficient when $k \ll N$.
- **Java `Set.add()` Idiom:** In Java, `set.add(x)` returns `false` if `x` was already present. Checking `if (!window.add(nums[i])) return true;` combines lookup and insertion into a single operation.