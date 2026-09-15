---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 2260: Minimum Consecutive Cards to Pick Up"
tags:
  - leetcode
  - coding
  - sliding-window
  - array
  - hash-table
  - amazon
  - google
---

# LeetCode 2260: Minimum Consecutive Cards to Pick Up

**Target Companies:** Amazon, Google, Microsoft  
**Difficulty:** Medium  
**Topic:** Sliding Window / Hash Map / Last-Seen Position Tracking  

---

### Problem Statement

You are given an integer array `cards` where `cards[i]` represents the value of the $i$-th card. A pair of cards are **matching** if the cards have the **same** value.

Return *the **minimum** number of consecutive cards you have to pick up to have a pair of matching cards among the picked cards*. If it is impossible to have matching cards, return `-1`.

---

### Input & Output Formats & Constraints

- **Input:** `cards: List[int]`
- **Output:** `int` (minimum length of a subarray containing a duplicate, or `-1`)
- **Constraints:**
  - $1 \le \text{cards.length} \le 10^5$
  - $0 \le \text{cards}[i] \le 10^6$

---

### Key Idea & Intuition

- **Closest Pair Principle:**
  - The problem asks for the minimum length of a contiguous subarray that contains at least two equal elements:
    $$\min_{i < j, \, \text{cards}[i] == \text{cards}[j]} (j - i + 1)$$
  - For any element that appears multiple times (e.g. at indices $i_1 < i_2 < i_3$), the distance between any non-adjacent occurrences is strictly greater than the distance between adjacent occurrences:
    $$i_3 - i_1 > i_2 - i_1 \quad \text{and} \quad i_3 - i_1 > i_3 - i_2$$
  - Therefore, we only ever need to compute distances between **consecutive appearances of the same card**!
- **Last-Seen Index Map:**
  - Maintain a hash map `last_seen` mapping `card_value -> most_recent_index`.
  - As we iterate through index $i$:
    - If `cards[i]` has been seen before:
      $$\text{length} = i - \text{last\_seen}[\text{cards}[i]] + 1$$
      $$\text{min\_len} = \min(\text{min\_len}, \text{length})$$
    - Update `last_seen[cards[i]] = i`.
- **Early Exit:**
  - If at any point $\text{min\_len} == 2$ (adjacent duplicate cards `[..., 5, 5, ...]`), we can immediately terminate and return `2`, since length 2 is the absolute minimum possible.

---

### Solution Approach (Step-by-Step)

1. Initialize `last_seen = {}` and `min_len = float('inf')`.
2. Loop `i` from $0$ to $\text{len}(cards) - 1$:
   - Let `card = cards[i]`.
   - If `card in last_seen`:
     - `min_len = min(min_len, i - last_seen[card] + 1)`
     - If `min_len == 2`: return `2` (cannot do better than adjacent cards).
   - Update `last_seen[card] = i`.
3. Return `-1` if `min_len == float('inf')` else `min_len`.

---

### Visual Algorithm Walkthrough

Let `cards = [3, 4, 2, 3, 4, 7]`.

```
Indices:   0  1  2  3  4  5
Cards:    [3, 4, 2, 3, 4, 7]

i=0, card=3: last_seen = {3: 0}
i=1, card=4: last_seen = {3: 0, 4: 1}
i=2, card=2: last_seen = {3: 0, 4: 1, 2: 2}

i=3, card=3:
- Found 3 in last_seen! Previous index = 0.
- Distance = 3 - 0 + 1 = 4. Subarray: [3, 4, 2, 3] (length 4).
- min_len = 4.
- Update last_seen[3] = 3.

i=4, card=4:
- Found 4 in last_seen! Previous index = 1.
- Distance = 4 - 1 + 1 = 4. Subarray: [4, 2, 3, 4] (length 4).
- min_len = min(4, 4) = 4.
- Update last_seen[4] = 4.

i=5, card=7:
- last_seen[7] = 5.

Final Minimum Length: 4.
```

---

### Solved Examples with Multiple Inputs

| Test Case | `cards` | Matching Pairs Identified | Min Length | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Example 1** | `[3, 4, 2, 3, 4, 7]` | `3..3` (len 4), `4..4` (len 4) | 4 | `4` |
| **Adjacent Match** | `[1, 0, 5, 3, 3]` | `3, 3` at indices 3, 4 | 2 | `2` |
| **All Unique** | `[1, 2, 3, 4]` | None | $\infty$ | `-1` |
| **Multiple Dupes** | `[7, 0, 7, 7, 7]` | `7, 7` at indices 2, 3 | 2 | `2` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List, Dict

class Solution:
    def minimumCardPickup(self, cards: List[int]) -> int:
        """
        Finds the minimum consecutive cards to pick up to get a matching pair.
        Uses last-seen hash map in a single O(N) pass.
        """
        last_seen: Dict[int, int] = {}
        min_len = float('inf')

        for i, card in enumerate(cards):
            if card in last_seen:
                dist = i - last_seen[card] + 1
                if dist < min_len:
                    min_len = dist
                    if min_len == 2:
                        return 2  # Absolute minimum possible length

            last_seen[card] = i

        return -1 if min_len == float('inf') else min_len
```

#### C++17
```cpp
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <climits>

class Solution {
public:
    int minimumCardPickup(const std::vector<int>& cards) {
        std::unordered_map<int, int> last_seen;
        int min_len = INT_MAX;

        for (int i = 0; i < static_cast<int>(cards.size()); ++i) {
            int card = cards[i];
            auto it = last_seen.find(card);

            if (it != last_seen.end()) {
                int dist = i - it->second + 1;
                min_len = std::min(min_len, dist);
                if (min_len == 2) return 2;
            }

            last_seen[card] = i;
        }

        return min_len == INT_MAX ? -1 : min_len;
    }
};
```

#### Java
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public int minimumCardPickup(int[] cards) {
        Map<Integer, Integer> lastSeen = new HashMap<>();
        int minLen = Integer.MAX_VALUE;

        for (int i = 0; i < cards.length; i++) {
            int card = cards[i];

            if (lastSeen.containsKey(card)) {
                int dist = i - lastSeen.get(card) + 1;
                minLen = Math.min(minLen, dist);
                if (minLen == 2) {
                    return 2;
                }
            }

            lastSeen.put(card, i);
        }

        return minLen == Integer.MAX_VALUE ? -1 : minLen;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ where $N = \text{cards.length}$.
  - We iterate through `cards` once.
  - Hash map lookup, insert, and update execute in $\mathcal{O}(1)$ average time.
  - Early return on `min_len == 2` provides further speedup.
  - Finishes in $< 20 \text{ ms}$ for $N = 10^5$.
- **Space Complexity:** $\mathcal{O}(\min(N, U))$ where $U$ is the number of unique card values in `cards` ($U \le 10^5$).

---

### Takeaway Pattern & Interview Traps

- **Consecutive Occurrence Optimization:** Never store a list of all indices for every number (`map[card] = [i1, i2, ...]`). Storing only the most recent index (`last_seen[card]`) guarantees optimal $\mathcal{O}(1)$ memory per unique card and linear runtime.
- **Early Termination at 2:** Length 2 is the absolute theoretical minimum for a pair of duplicate cards. Exiting immediately when `min_len == 2` is a clean micro-optimization.