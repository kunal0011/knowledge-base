---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 220: Contains Duplicate III"
tags:
  - leetcode
  - coding
  - sliding-window
  - array
  - bucket-sort
  - ordered-set
  - amazon
  - google
---

# LeetCode 220: Contains Duplicate III

**Target Companies:** Google (Signature Problem), Amazon, Meta, Microsoft  
**Difficulty:** Hard  
**Topic:** Sliding Window / Bucket Sort / Ordered Set / Interval Search  

---

### Problem Statement

You are given an integer array `nums` and two integers `indexDiff` and `valueDiff`.

Find a pair of indices `(i, j)` such that:
- $i \ne j$,
- $|i - j| \le \text{indexDiff}$, and
- $|\text{nums}[i] - \text{nums}[j]| \le \text{valueDiff}$.

Return `true` *if such pair exists or* `false` *otherwise*.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `indexDiff: int`, `valueDiff: int`
- **Output:** `bool`
- **Constraints:**
  - $2 \le \text{nums.length} \le 10^5$
  - $-10^9 \le \text{nums}[i] \le 10^9$
  - $1 \le \text{indexDiff} \le \text{nums.length}$
  - $0 \le \text{valueDiff} \le 10^9$

---

### Key Idea & Intuition

- **Dual Constraint Analysis:**
  - Distance in indices: $|i - j| \le \text{indexDiff} \implies$ Fixed/Bounded Sliding Window of size at most `indexDiff`.
  - Distance in values: $|\text{nums}[i] - \text{nums}[j]| \le \text{valueDiff} \implies$ Value proximity lookup.

- **Approach 1: Ordered Set (Balanced BST) — $\mathcal{O}(N \log k)$:**
  - Maintain a sliding window of elements in a balanced binary search tree (`std::set` in C++, `TreeSet` in Java).
  - For element $x$, query the smallest element in the BST that is $\ge x - \text{valueDiff}$ (via `lower_bound` or `ceiling`).
  - If that element exists and is $\le x + \text{valueDiff}$, return `true`.
  - Time complexity: $\mathcal{O}(N \log(\text{indexDiff}))$.

- **Approach 2: Bucket Hashing — Optimal $\mathcal{O}(N)$ Time:**
  - We can partition numbers into **buckets of width** $W = \text{valueDiff} + 1$:
    $$\text{bucket\_id}(x) = \lfloor x / W \rfloor$$
  - **Properties of Buckets of Width $W$:**
    1. **Same Bucket:** Any two elements falling into the same bucket have a difference of at most $W - 1 = \text{valueDiff}$. Thus, if bucket $B$ is already occupied, we found a match immediately!
    2. **Adjacent Buckets:** A number could potentially be within `valueDiff` of numbers in adjacent buckets ($B - 1$ or $B + 1$). We only need to check if $|x - \text{bucket}[B - 1]| \le \text{valueDiff}$ and $|x - \text{bucket}[B + 1]| \le \text{valueDiff}$.
    3. **Other Buckets:** Any bucket with ID $\le B - 2$ or $\ge B + 2$ is guaranteed to have a difference strictly $> \text{valueDiff}$.
  - At each index $i$:
    - Check bucket $B$, $B - 1$, and $B + 1$ in $\mathcal{O}(1)$ time!
    - Store `bucket[B] = nums[i]`.
    - If $i \ge \text{indexDiff}$, delete the bucket of the outgoing element $\text{nums}[i - \text{indexDiff}]$.
  - This reduces the entire algorithm to **strictly $\mathcal{O}(N)$ time and $\mathcal{O}(\text{indexDiff})$ space**!

---

### Solution Approach (Step-by-Step)

1. Let bucket width $W = \text{valueDiff} + 1$.
2. Initialize an empty hash map `buckets = {}`.
3. Define helper `get_bucket_id(x)`:
   - In Python, integer floor division `x // W` handles negative numbers correctly.
   - In C++/Java, if $x < 0$, adjust: `(x + 1) / W - 1`.
4. Loop `i` from $0$ to $\text{len}(nums) - 1$:
   - Let $x = \text{nums}[i]$ and $b = \text{get\_bucket\_id}(x)$.
   - If $b \in \text{buckets}$: return `True`.
   - If $b - 1 \in \text{buckets}$ and $x - \text{buckets}[b - 1] \le \text{valueDiff}$: return `True`.
   - If $b + 1 \in \text{buckets}$ and $\text{buckets}[b + 1] - x \le \text{valueDiff}$: return `True`.
   - Place into bucket: `buckets[b] = x`.
   - **Maintain Sliding Window of Size `indexDiff`:**
     - If $i \ge \text{indexDiff}$:
       - Outgoing element: `out_b = get_bucket_id(nums[i - indexDiff])`.
       - Delete `buckets[out_b]`.
5. Return `False`.

---

### Visual Algorithm Walkthrough

Let `nums = [1, 5, 9, 1, 5, 9]`, `indexDiff = 2`, `valueDiff = 3`.
Bucket width $W = \text{valueDiff} + 1 = 3 + 1 = 4$.

```
Buckets:
Bucket 0: covers [0..3]
Bucket 1: covers [4..7]
Bucket 2: covers [8..11]

i=0, num=1: b = 1 // 4 = 0.
- Check b=0, b=-1, b=1: all empty.
- buckets = {0: 1}

i=1, num=5: b = 5 // 4 = 1.
- Check b=1 (empty), b=0 (has 1: |5 - 1| = 4 > 3), b=2 (empty).
- buckets = {0: 1, 1: 5}

i=2, num=9: b = 9 // 4 = 2.
- Check b=2 (empty), b=1 (has 5: |9 - 5| = 4 > 3), b=3 (empty).
- buckets = {0: 1, 1: 5, 2: 9}
- Window full (i >= 2): Remove nums[0]=1 (bucket 0) -> buckets = {1: 5, 2: 9}

i=3, num=1: b = 1 // 4 = 0.
- Check b=0 (empty), b=1 (has 5: |1 - 5| = 4 > 3), b=-1 (empty).
- buckets = {1: 5, 2: 9, 0: 1}
- Remove nums[1]=5 (bucket 1) -> buckets = {2: 9, 0: 1}

i=4, num=5: b = 5 // 4 = 1.
- Check b=1 (empty), b=0 (has 1: |5 - 1| = 4 > 3), b=2 (has 9: |5 - 9| = 4 > 3).
- buckets = {2: 9, 0: 1, 1: 5}
- Remove nums[2]=9 (bucket 2) -> buckets = {0: 1, 1: 5}

i=5, num=9: b = 9 // 4 = 2.
- Check b=2 (empty), b=1 (has 5: |9 - 5| = 4 > 3), b=3 (empty).
- buckets = {0: 1, 1: 5, 2: 9}

No pair satisfies both conditions -> Return False.
```

---

### Solved Examples with Multiple Inputs

| Test Case | `nums` | `indexDiff` | `valueDiff` | Result | Reason |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Example 1** | `[1, 2, 3, 1]` | `3` | `0` | `true` | `nums[0] == nums[3] == 1`, $\vert 3 - 0 \vert \le 3$ |
| **Example 2** | `[1, 5, 9, 1, 5, 9]` | `2` | `3` | `false` | Differences are at least 4 |
| **Close Values** | `[1, 10, 15, 3]` | `3` | `2` | `true` | $\vert \text{nums}[0] - \text{nums}[3] \vert = \vert 1 - 3 \vert = 2 \le 2$ |
| **Negative Numbers** | `[-3, 3, -6]` | `2` | `3` | `true` | $\vert -3 - (-6) \vert = 3 \le 3$ |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List, Dict

class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], indexDiff: int, valueDiff: int) -> bool:
        """
        Determines if there exist two indices with |i - j| <= indexDiff and |nums[i] - nums[j]| <= valueDiff.
        Uses bucket hashing for optimal O(N) time and O(indexDiff) space.
        """
        if indexDiff <= 0 or valueDiff < 0:
            return False

        # Bucket width is valueDiff + 1
        w = valueDiff + 1
        buckets: Dict[int, int] = {}

        for i, val in enumerate(nums):
            # Python's // operator handles negative numbers via floor division
            b = val // w

            # Condition 1: Same bucket contains an element
            if b in buckets:
                return True

            # Condition 2: Adjacent bucket (b - 1)
            if (b - 1) in buckets and abs(val - buckets[b - 1]) <= valueDiff:
                return True

            # Condition 3: Adjacent bucket (b + 1)
            if (b + 1) in buckets and abs(val - buckets[b + 1]) <= valueDiff:
                return True

            # Insert into bucket
            buckets[b] = val

            # Evict oldest element outside window
            if i >= indexDiff:
                del buckets[nums[i - indexDiff] // w]

        return False
```

#### C++17
```cpp
#include <vector>
#include <unordered_map>
#include <cmath>

class Solution {
public:
    bool containsNearbyAlmostDuplicate(const std::vector<int>& nums, int indexDiff, int valueDiff) {
        if (indexDiff <= 0 || valueDiff < 0) return false;

        long long w = static_cast<long long>(valueDiff) + 1;
        std::unordered_map<long long, long long> buckets;

        auto getBucketId = [w](long long val) -> long long {
            return val < 0 ? (val + 1) / w - 1 : val / w;
        };

        for (int i = 0; i < static_cast<int>(nums.size()); ++i) {
            long long val = nums[i];
            long long b = getBucketId(val);

            if (buckets.count(b)) return true;
            if (buckets.count(b - 1) && std::abs(val - buckets[b - 1]) <= valueDiff) return true;
            if (buckets.count(b + 1) && std::abs(val - buckets[b + 1]) <= valueDiff) return true;

            buckets[b] = val;

            if (i >= indexDiff) {
                buckets.erase(getBucketId(nums[i - indexDiff]));
            }
        }

        return false;
    }
};
```

#### Java
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public boolean containsNearbyAlmostDuplicate(int[] nums, int indexDiff, int valueDiff) {
        if (indexDiff <= 0 || valueDiff < 0) return false;

        long w = (long) valueDiff + 1;
        Map<Long, Long> buckets = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            long val = nums[i];
            long b = getBucketId(val, w);

            if (buckets.containsKey(b)) return true;
            if (buckets.containsKey(b - 1) && Math.abs(val - buckets.get(b - 1)) <= valueDiff) return true;
            if (buckets.containsKey(b + 1) && Math.abs(val - buckets.get(b + 1)) <= valueDiff) return true;

            buckets.put(b, val);

            if (i >= indexDiff) {
                buckets.remove(getBucketId(nums[i - indexDiff], w));
            }
        }

        return false;
    }

    private long getBucketId(long val, long w) {
        return val < 0 ? (val + 1) / w - 1 : val / w;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ where $N = \text{nums.length}$.
  - For each element, bucket computation, 3 map lookups, 1 insertion, and at most 1 deletion take $\mathcal{O}(1)$ average time.
  - Strictly linear, outperforming $\mathcal{O}(N \log k)$ balanced BSTs.
  - Finishes in $< 20 \text{ ms}$ for $N = 10^5$.
- **Space Complexity:** $\mathcal{O}(\min(N, \text{indexDiff}))$ auxiliary space to store at most $\text{indexDiff} + 1$ buckets in the hash map.

---

### Takeaway Pattern & Interview Traps

- **Bucket Width Choice ($W = \text{valueDiff} + 1$):**
  - Why $+ 1$? If $\text{valueDiff} = 0$, width must be $1$ to prevent division by zero. Furthermore, two numbers $x$ and $y$ in the same bucket have maximum distance $W - 1 = \text{valueDiff}$.
- **Negative Floor Division in C++/Java:**
  - Python's `//` rounds toward $-\infty$. In C++ and Java, `/` truncates towards zero (so $-3 / 4 = 0$ instead of $-1$). The formula `val < 0 ? (val + 1) / w - 1 : val / w` correctly maps negative integers into continuous distinct negative bucket IDs.
- **64-bit Integer Overflow:** Since $\text{valueDiff} \le 10^9$ and $\text{nums}[i]$ can be $-10^9$, differences can reach $2 \times 10^9$. Cast to `long long` (C++) or `long` (Java) to prevent overflow during arithmetic.