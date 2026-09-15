---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 881: Boats to Save People"
tags:
  - leetcode
  - coding
  - greedy
  - two-pointers
  - sorting
  - array
  - amazon
  - google
  - uber
---

# LeetCode 881: Boats to Save People

**Target Companies:** Amazon, Google, Uber, Salesforce, Apple  
**Difficulty:** Medium  
**Topic:** Greedy / Two Pointers / Sorting  

---

### Problem Statement

You are given an array `people` where `people[i]` is the weight of the $i^{\text{th}}$ person, and an integer `limit` representing the maximum weight a boat can carry. Each boat can carry at most **two people** at the same time, provided the sum of the weight of those people is at most `limit`.

Return *the minimum number of boats to carry every given person*.

---

### Input & Output Formats & Constraints

- **Input:**
  - An integer array `people` ($1 \le |people| \le 5 \times 10^4$).
  - An integer `limit` ($1 \le \text{limit} \le 3 \times 10^4$).
- **Output:** An integer denoting the minimum number of rescue boats required.
- **Constraints:**
  - `1 <= people.length <= 5 * 10^4`
  - `1 <= people[i] <= limit <= 3 * 10^4`
  - Maximum boat capacity is strictly **2 people**.

---

### Key Idea & Intuition

#### The Heaviest-Person Dilemma
The hardest person to accommodate is always the heaviest remaining person.
Because a boat can carry at most **two people**:
1. The heaviest person at index `right` must take a boat.
2. If this heavy person can be paired with anyone at all, their best chance is to pair with the **lightest remaining person** at index `left`.
3. If `people[left] + people[right] <= limit`, then:
   - They successfully share a boat.
   - We advance `left` and decrement `right`.
4. If `people[left] + people[right] > limit`, then:
   - The heaviest person cannot pair with the lightest person.
   - Because all other remaining people are even heavier than `people[left]`, the person at `right` **cannot pair with anyone remaining**.
   - Consequently, the heavy person must ride alone in their own boat!
   - We decrement `right` only.

#### Greedy Invariant & Optimality
Pairing the heaviest person with the lightest person whenever feasible saves the maximum remaining capacity for other pairs. Placing the heaviest person alone when they cannot even pair with the lightest person is forced and unavoidable. Hence, this greedy two-pointer approach is provably optimal.

---

### Solution Approach (Step-by-Step)

1. **Sort the Weights:**
   - Sort the array `people` in ascending order.
2. **Initialize Two Pointers:**
   - `left = 0` (lightest person)
   - `right = len(people) - 1` (heaviest person)
   - `boats = 0`
3. **Two-Pointer Traversal:**
   - While `left <= right`:
     - If `people[left] + people[right] <= limit`:
       - Pair them: `left += 1`
     - The heaviest person always takes a boat: `right -= 1`
     - Increment `boats += 1`
4. **Return:**
   - Return `boats`.

---

### Visual Algorithm Walkthrough

#### Trace for `people = [3, 2, 2, 1]`, `limit = 3`
```
Step 1: Sort array
people = [1, 2, 2, 3], limit = 3
Pointers: left = 0 (val 1), right = 3 (val 3)

Round 1:
- Check people[left] + people[right] = 1 + 3 = 4 > 3 (limit).
- Person 3 CANNOT pair with person 1.
- Since 1 is the lightest, person 3 cannot pair with anyone.
- Person 3 rides alone: Boat 1 = [3].
- right becomes 2, boats = 1.

Round 2:
- left = 0 (val 1), right = 2 (val 2)
- Check people[left] + people[right] = 1 + 2 = 3 <= 3 (limit).
- Valid pair!
- Boat 2 = [1, 2].
- left becomes 1, right becomes 1, boats = 2.

Round 3:
- left = 1 (val 2), right = 1 (val 2)
- left == right -> Only one person remaining.
- Person 2 rides alone: Boat 3 = [2].
- right becomes 0, boats = 3.

Loop terminates (left > right).
Total boats = 3.
```

---

### Solved Examples with Multiple Inputs

| Input `people` | `limit` | Sorted Array | Step-by-Step Boat Allocations | Output |
|---|---|---|---|---|
| `[1, 2]` | `3` | `[1, 2]` | $1 + 2 = 3 \le 3 \to [1, 2]$ | `1` |
| `[3, 2, 2, 1]` | `3` | `[1, 2, 2, 3]` | $[3]$, $[1, 2]$, $[2]$ | `3` |
| `[3, 5, 3, 4]` | `5` | `[3, 3, 4, 5]` | $[5]$, $[3, 4] \to [4]$, $[3, 3] \to [3, 3]$, $[3]$ | `4` |
| `[5, 1, 4, 2]` | `6` | `[1, 2, 4, 5]` | $[1, 5]$, $[2, 4]$ | `2` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def numRescueBoats(self, people: list[int], limit: int) -> int:
        people.sort()
        
        left: int = 0
        right: int = len(people) - 1
        boats: int = 0
        
        while left <= right:
            # If lightest and heaviest can share a boat, pair them
            if people[left] + people[right] <= limit:
                left += 1
            # Heaviest person always consumes a boat
            right -= 1
            boats += 1
            
        return boats
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int numRescueBoats(std::vector<int>& people, int limit) {
        std::sort(people.begin(), people.end());
        
        int left = 0;
        int right = static_cast<int>(people.size()) - 1;
        int boats = 0;
        
        while (left <= right) {
            if (people[left] + people[right] <= limit) {
                left++;
            }
            // Heaviest person takes this boat
            right--;
            boats++;
        }
        
        return boats;
    }
};
```

#### Java 17
```java
import java.util.Arrays;

class Solution {
    public int numRescueBoats(int[] people, int limit) {
        Arrays.sort(people);
        
        int left = 0;
        int right = people.length - 1;
        int boats = 0;
        
        while (left <= right) {
            if (people[left] + people[right] <= limit) {
                left++;
            }
            // Heaviest passenger always boarded
            right--;
            boats++;
        }
        
        return boats;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \log N)$, where $N = |people|$. Sorting the array takes $\mathcal{O}(N \log N)$ time, followed by a linear two-pointer scan taking $\mathcal{O}(N)$ time.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space if sorting is done in-place (or $\mathcal{O}(N)$ / $\mathcal{O}(\log N)$ depending on language sorting runtime implementation like Timsort in Python or Dual-Pivot Quicksort in Java).

---

### Takeaway Pattern & Interview Traps

1. **At Most Two Passengers Constraint:** The two-pointer greedy strategy depends critically on the constraint that a boat can hold **at most 2 people**. If a boat could hold an arbitrary number of passengers up to `limit`, this would become the NP-hard Bin Packing Problem!
2. **Always Decrement `right`:** Since every iteration assigns a boat to `people[right]`, `right` decrements unconditionally. Only `left` increments conditionally.
3. **Equal Pointers (`left == right`):** When one person is left, the condition `people[left] + people[right] <= limit` evaluates `2 * people[left] <= limit`. Even if true, both pointers decrement past each other, safely counting exactly one boat.