---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 1338: Reduce Array Size to The Half"
tags:
  - leetcode
  - coding
  - greedy
  - hash-table
  - sorting
  - bucket-sort
  - amazon
  - google
---

# LeetCode 1338: Reduce Array Size to The Half

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Hash Table / Sorting / Bucket Sort  

---

### Problem Statement

You are given an integer array `arr`. You can choose a set of integers and remove all the occurrences of these integers in the array.

Return the **minimum size of the set** so that at least half of the integers of the array are removed.

---

### Input & Output Formats & Constraints

- **Input:**
  - `arr`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{arr.length} \le 10^5$).
- **Output:**
  - `int` — the minimum number of distinct integers removed such that total removed count $\ge \lfloor \frac{n + 1}{2} \rfloor$ (or remaining $\le \frac{n}{2}$).
- **Constraints:**
  - $2 \le \text{arr.length} \le 10^5$ (guaranteed to be even).
  - $1 \le \text{arr}[i] \le 10^5$

---

### Key Idea & Intuition

Every operation allows choosing one unique integer value and removing **all** of its occurrences from the array.
- Each choice costs **exactly 1 set element**.
- Choosing integer $x$ eliminates $\text{count}(x)$ elements from the array.

To reach the target of removing at least $n / 2$ elements using the minimum number of set elements, we must greedily pick integers that yield the largest reduction per operation:
> **Always pick the integer with the largest remaining frequency.**

#### Implementations:
1. **Sorting Frequencies ($\mathcal{O}(n \log n)$ time):** Count frequencies using a hash map, sort the frequency counts descending, and accumulate until the sum reaches $\ge n / 2$.
2. **Bucket Sort Optimization ($\mathcal{O}(n)$ time, $\mathcal{O}(n)$ space):** Since the frequency of any integer cannot exceed $n$, we can place frequencies into an array of buckets `buckets[f]`, representing how many distinct integers have frequency $f$. We then iterate $f$ downwards from $n$ to $1$, collecting items in strictly linear $\mathcal{O}(n)$ time without comparison-based sorting!

---

### Solution Approach (Step-by-Step: Bucket Sort $\mathcal{O}(n)$)

1. Compute frequency map `counts` of `arr`.
2. Let $n = \text{len}(arr)$ and `target = n // 2`.
3. Create bucket array `buckets` of size $n + 1$, where `buckets[count]` is the number of distinct values with that frequency.
4. For each frequency `f` in `counts.values()`:
   - `buckets[f] += 1`
5. Initialize `removed = 0`, `set_size = 0`, and loop `f` from $n$ down to $1$:
   - While `buckets[f] > 0`:
     - Determine how many numbers of frequency $f$ we need:
       `take = min(buckets[f], (target - removed + f - 1) // f)`
     - Update `removed += take * f` and `set_size += take`.
     - `buckets[f] -= take`
     - If `removed >= target`: return `set_size`.
6. Return `set_size`.

---

### Visual Algorithm Walkthrough

For `arr = [3, 3, 3, 3, 5, 5, 5, 2, 2, 7]` ($n = 10$, target to remove $\ge 5$):

```
Frequencies:
  3 -> 4
  5 -> 3
  2 -> 2
  7 -> 1

Sorted Frequencies: [4, 3, 2, 1]

Step 1: Pick '3' (frequency = 4)
  removed = 4 < 5
  set_size = 1

Step 2: Pick '5' (frequency = 3)
  removed = 4 + 3 = 7 >= 5 (Target satisfied!)
  set_size = 2

Minimum set size = 2 (e.g. set {3, 5}).
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `arr = [3, 3, 3, 3, 5, 5, 5, 2, 2, 7]`
- **Output:** `2`

#### Example 2:
- **Input:** `arr = [7, 7, 7, 7, 7, 7]`
- **Tracing:** Single value 7 has frequency 6. Pick {7}, removed = 6 $\ge 3$.
- **Output:** `1`

#### Example 3:
- **Input:** `arr = [1, 9]`
- **Tracing:** Frequencies [1, 1]. Pick either 1 or 9, removed = 1 $\ge 1$.
- **Output:** `1`

#### Example 4:
- **Input:** `arr = [1000, 1000, 3, 7]`
- **Tracing:** Frequency 2 (for 1000) achieves $\ge 4 / 2 = 2$.
- **Output:** `1`

---

### Multi-Language Implementations

#### Python 3 (Optimal $\mathcal{O}(n)$ Bucket Sort)
```python
from collections import Counter
from typing import List

class Solution:
    def minSetSize(self, arr: List[int]) -> int:
        n = len(arr)
        target = (n + 1) // 2
        counts = Counter(arr)
        
        # Bucket sort by frequency: buckets[f] = how many unique numbers have frequency f
        buckets = [0] * (n + 1)
        for count in counts.values():
            buckets[count] += 1
            
        removed = 0
        set_size = 0
        
        # Traverse frequencies from largest down to 1
        for f in range(n, 0, -1):
            if buckets[f] == 0:
                continue
                
            needed = (target - removed + f - 1) // f
            take = min(buckets[f], needed)
            
            removed += take * f
            set_size += take
            
            if removed >= target:
                return set_size
                
        return set_size
```

#### C++17 (Optimal $\mathcal{O}(n)$ Bucket Sort)
```cpp
#include <vector>
#include <unordered_map>
#include <algorithm>

class Solution {
public:
    int minSetSize(const std::vector<int>& arr) {
        int n = static_cast<int>(arr.size());
        int target = (n + 1) / 2;
        
        std::unordered_map<int, int> counts;
        for (int x : arr) {
            counts[x]++;
        }
        
        std::vector<int> buckets(n + 1, 0);
        for (const auto& [val, count] : counts) {
            buckets[count]++;
        }
        
        int removed = 0;
        int set_size = 0;
        
        for (int f = n; f >= 1; --f) {
            if (buckets[f] == 0) continue;
            
            int needed = (target - removed + f - 1) / f;
            int take = std::min(buckets[f], needed);
            
            removed += take * f;
            set_size += take;
            
            if (removed >= target) {
                return set_size;
            }
        }
        
        return set_size;
    }
};
```

#### Java 17 (Optimal $\mathcal{O}(n)$ Bucket Sort)
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public int minSetSize(int[] arr) {
        int n = arr.length;
        int target = (n + 1) / 2;
        
        Map<Integer, Integer> counts = new HashMap<>();
        for (int num : arr) {
            counts.put(num, counts.getOrDefault(num, 0) + 1);
        }
        
        int[] buckets = new int[n + 1];
        for (int count : counts.values()) {
            buckets[count]++;
        }
        
        int removed = 0;
        int setSize = 0;
        
        for (int f = n; f >= 1; f--) {
            if (buckets[f] == 0) continue;
            
            int needed = (target - removed + f - 1) / f;
            int take = Math.min(buckets[f], needed);
            
            removed += take * f;
            setSize += take;
            
            if (removed >= target) {
                return setSize;
            }
        }
        
        return setSize;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - Counting frequencies with hash map takes $\mathcal{O}(n)$ time.
  - Bucket sort frequency aggregation takes $\mathcal{O}(U) \le \mathcal{O}(n)$ where $U$ is distinct numbers.
  - Sweeping bucket indices from $n$ down to $1$ takes $\mathcal{O}(n)$ steps.
  - Total runtime is strictly linear $\mathcal{O}(n)$.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space
  - The frequency hash map and `buckets` array both take $\mathcal{O}(n)$ memory.

---

### Takeaway Pattern & Interview Traps

- **Linear Sorting with Bounded Frequencies:** Whenever sorting frequencies of elements from an array of size $n$, the frequencies are bounded within $[1, n]$. Utilizing **Bucket Sort** drops time complexity from $\mathcal{O}(n \log n)$ to pure $\mathcal{O}(n)$.
- **Strict Integer Rounding:** Notice `needed = (target - removed + f - 1) // f`. Using integer ceiling division avoids floating point overhead and ensures exactly the correct number of items are taken.