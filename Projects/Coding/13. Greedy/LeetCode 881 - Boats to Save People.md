---
date: "2025-12-18"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 881: Boats to Save People"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 881: Boats to Save People

Below is a **complete, structured, and exam-ready explanation** of **LeetCode 881 – Boats to Save People**, aligned with how greedy problems are typically reasoned in interviews and competitive programming.

---

## LeetCode 881: Boats to Save People

### Problem Statement

You are given an array `people` where `people[i]` is the weight of the `i`-th person, and an integer `limit` representing the maximum weight a boat can carry.

Each boat can carry **at most two people**, provided their combined weight is **less than or equal to `limit`**.

**Return the minimum number of boats required** to carry every person.

---

### Key Observations

1. **Each boat carries at most two people**

   * Either:

     * One heavy person alone
     * One heavy + one light person
2. **Heaviest people are the hardest to place**

   * If the heaviest person cannot pair with the lightest person, they cannot pair with *anyone*.
3. **Local optimal choice leads to global optimal solution**

   * Pair the **lightest possible person** with the **heaviest possible person** whenever feasible.
4. This naturally leads to:

   * **Sorting**
   * **Two-pointer greedy strategy**

---

### Greedy Strategy (Core Idea)

1. Sort the array of weights.
2. Use two pointers:

   * `left` → lightest person
   * `right` → heaviest person
3. Always try to put:

   * `people[left] + people[right]` in one boat
4. If they fit:

   * Move both pointers inward
5. If they don’t:

   * Heaviest person goes alone
   * Move only `right`
6. Each step uses exactly **one boat**

---

### Why This Greedy Works

* If the **heaviest** person cannot pair with the **lightest**, they cannot pair with anyone else.
* Pairing the heaviest with the lightest (when possible) **maximizes utilization** of each boat.
* This avoids wasting capacity and minimizes the total boats.

---

### Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        people.sort()
        
        left: int = 0
        right: int = len(people) - 1
        boats: int = 0
        
        while left <= right:
            # Try to pair the lightest with the heaviest
            if people[left] + people[right] <= limit:
                left += 1  # lightest person is used
            
            # Heaviest person always uses a boat
            right -= 1
            boats += 1
        
        return boats
```

---

### Complete Worked Example (Step-by-Step)

#### Input

```
people = [3, 2, 2, 1]
limit = 3
```

---

#### Step 1: Sort the array

```
people = [1, 2, 2, 3]
```

| Pointer | Index | Weight |
| --- | --- | --- |
| left | 0 | 1 |
| right | 3 | 3 |

---

### Step 2: Greedy Pairing Process

#### Iteration 1

* `1 + 3 = 4` ❌ (exceeds limit)
* Heaviest (3) goes alone

```
Boat 1 → [3]
```

Pointers:

```
left = 0
right = 2
boats = 1
```

---

#### Iteration 2

* `1 + 2 = 3` ✅ (fits)
* Pair them

```
Boat 2 → [1, 2]
```

Pointers:

```
left = 1
right = 1
boats = 2
```

---

#### Iteration 3

* Only one person left (`2`)
* Goes alone

```
Boat 3 → [2]
```

Pointers:

```
left = 1
right = 0 (loop ends)
boats = 3
```

---

### Final Answer

```
Minimum boats required = 3
```

---

### Time and Space Complexity

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n log n)** (sorting) |
| Space Complexity | **O(1)** extra space (in-place sorting) |

---

### Interview Takeaways

* This is a **classic greedy + two-pointer** problem.
* Key skill tested:

  * Recognizing that **pairing extremes** is optimal.
  * Understanding **why greedy is safe** here.
* Similar problems:

  * Two Sum variants
  * Interval pairing
  * Load balancing

---

If you want, I can also:

* Show a **greedy proof**
* Compare with **incorrect greedy approaches**
* Draw a **decision tree of pointer movement**
* Relate this to **bin packing approximation logic**

Just tell me.