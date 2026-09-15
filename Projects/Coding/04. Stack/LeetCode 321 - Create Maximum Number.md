---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 321: Create Maximum Number"
tags:
  - leetcode
  - coding
  - stack
---

# LeetCode 321: Create Maximum Number

Below is a **complete, interview-ready explanation** of **LeetCode 321 – Create Maximum Number**, structured exactly as requested.

---

## LeetCode 321: Create Maximum Number

### 🔹 Problem Statement

You are given two integer arrays `nums1` and `nums2` of lengths `m` and `n`, respectively, and an integer `k`.

Create the **maximum possible number of length `k`** by selecting digits from `nums1` and `nums2`, while preserving the **relative order** of digits from each array.

Return the resulting number as an array of digits.

---

### 🔹 Key Observations

1. **Order must be preserved**

   * This is **not permutation**; it is **subsequence selection**.
2. **We must choose digits from both arrays**

   * If we choose `i` digits from `nums1`, then we must choose `k - i` digits from `nums2`.
3. **Greedy choice per array**

   * From a single array, selecting the largest subsequence of length `t` is a **monotonic stack** problem.
4. **Merge is lexicographic**

   * When merging two subsequences, at each step we choose the lexicographically larger remaining suffix.
5. **Brute force split is bounded**

   * Valid splits:

     ```
     max(0, k - len(nums2)) ≤ i ≤ min(k, len(nums1))
     ```

---

### 🔹 Stack Key Insight (Very Important)

**Problem reduced to two sub-problems:**

#### 1. Maximum subsequence of length `t` from one array

This is solved using a **monotonic decreasing stack**.

**Rule**:

* While:

  * stack is not empty
  * current digit > stack top
  * we can still remove digits
* → pop stack

This ensures:

* Larger digits appear earlier
* Order is preserved

---

#### 2. Merge two subsequences

At each step:

* Compare remaining suffixes lexicographically
* Pick the larger one’s first digit

Python makes this easy using list comparison.

---

### 🔹 Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:

        # Monotonic stack to get max subsequence of length t
        def max_subsequence(nums: List[int], t: int) -> List[int]:
            stack = []
            drop = len(nums) - t

            for num in nums:
                while drop > 0 and stack and stack[-1] < num:
                    stack.pop()
                    drop -= 1
                stack.append(num)

            return stack[:t]

        # Merge two subsequences greedily
        def merge(a: List[int], b: List[int]) -> List[int]:
            result = []
            while a or b:
                if a > b:
                    result.append(a.pop(0))
                else:
                    result.append(b.pop(0))
            return result

        best = []

        # Try all valid splits
        for i in range(max(0, k - len(nums2)), min(k, len(nums1)) + 1):
            part1 = max_subsequence(nums1, i)
            part2 = max_subsequence(nums2, k - i)
            candidate = merge(part1[:], part2[:])
            best = max(best, candidate)

        return best
```

---

### 🔹 Worked Example (Step-by-Step)

#### Input

```
nums1 = [3, 4, 6, 5]
nums2 = [9, 1, 2, 5, 8, 3]
k = 5
```

---

### Step 1: Try all valid splits

Possible `(i, k-i)`:

```
i = 2 → (2 from nums1, 3 from nums2)
i = 3 → (3 from nums1, 2 from nums2)
i = 4 → (4 from nums1, 1 from nums2)
```

---

### Step 2: Example split → `i = 2`

#### Max subsequence from nums1 (length 2)

```
nums1 = [3,4,6,5]

stack evolution:
3 → [3]
4 → pop 3 → [4]
6 → pop 4 → [6]
5 → [6,5]

result → [6,5]
```

#### Max subsequence from nums2 (length 3)

```
nums2 = [9,1,2,5,8,3]

result → [9,8,3]
```

---

### Step 3: Merge `[6,5]` and `[9,8,3]`

| Compare | Pick |
| --- | --- |
| [6,5] vs [9,8,3] | 9 |
| [6,5] vs [8,3] | 8 |
| [6,5] vs [3] | 6 |
| [5] vs [3] | 5 |
| [] vs [3] | 3 |

Result:

```
[9,8,6,5,3]
```

---

### Step 4: Compare all candidates

Final maximum:

```
[9,8,6,5,3]
```

---

### 🔹 Time & Space Complexity

| Component | Complexity |
| --- | --- |
| Subsequence selection | O(n) |
| Merge | O(k²) worst (due to pop(0)) |
| Total | O(k · (m + n)) |
| Space | O(k) |

*(Can optimize merge using indices instead of pop)*

---

### 🔹 Core Interview Takeaways

* **This is NOT a single stack problem** — it is:

  ```
  Stack + Greedy + Lexicographic Merge
  ```
* Always split `k` across arrays
* Use **monotonic stack** for max subsequence
* Merge by **suffix comparison**, not digit comparison

---

If you want:

* Optimized merge (no `pop(0)`)
* Visual stack evolution diagrams
* Similar problems to practice (321 pattern)

Tell me how deep you want to go.