---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 406: Queue Reconstruction by Height"
tags:
  - leetcode
  - coding
  - greedy
  - amazon
  - google
---

# LeetCode 406: Queue Reconstruction by Height

**Target Companies:** Google (Signature Classic), Amazon, Meta  
**Difficulty:** Medium  
**Topic:** Greedy Sorting + List Insertion

---

### Problem Statement

You are given an array of people, `people`, which are the attributes of some people in a queue (not necessarily in order). Each `people[i] = [h_i, k_i]` represents the $i$-th person of height $h_i$ with **exactly** $k_i$ other people in front who have a height greater than or equal to $h_i$.

Reconstruct and return the queue that is represented by the input array `people`. The returned queue should be formatted as an array `queue`, where `queue[j] = [h_j, k_j]` is the attributes of the $j$-th person in the queue (`queue[0]` is the person at the front of the queue).

---

### Input & Output Formats & Constraints

- **Input:** `people: List[List[int]]` where `people[i] = [h, k]`
- **Output:** `List[List[int]]` representing the reconstructed queue.
- **Constraints:**
  - $1 \le \text{people.length} \le 2000$
  - $0 \le h_i \le 10^6$
  - $0 \le k_i < \text{people.length}$
  - It is guaranteed that the queue can be reconstructed.

---

### Key Idea & Intuition

- **Why Greedy Works (Taller People Don't See Shorter People):**
  - Shorter people do **NOT** affect the $k$-count of taller people!
  - Therefore, if we place all **taller people first**, placing a shorter person afterwards will never disrupt the count of any taller person already placed.
- **Sorting Criterion:**
  1. Primary: Sort height $h$ in **descending order** (tallest first).
  2. Secondary: If heights are equal, sort count $k$ in **ascending order** (so smaller $k$ is placed first).
- **Insertion Rule:**
  - Iterate through the sorted people. For each person `[h, k]`, simply insert them at index `k` (`result.insert(k, person)`)!
  - Because all people already in `result` are $\ge h$, placing `[h, k]` at index `k` guarantees that there are exactly $k$ taller or equal people ahead of them!

---

### Solution Approach (Step-by-Step)

1. Sort `people` using custom key: `(-h, k)`.
2. Initialize `queue = []`.
3. For each `person = [h, k]` in sorted `people`:
   - `queue.insert(k, person)`.
4. Return `queue`.

---

### Visual Algorithm Walkthrough

```
Original: [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]

Sorted by (-h, k):
1. [7, 0]
2. [7, 1]
3. [6, 1]
4. [5, 0]
5. [5, 2]
6. [4, 4]

Step-by-step Insertions:
Insert [7, 0] at 0: [[7, 0]]
Insert [7, 1] at 1: [[7, 0], [7, 1]]
Insert [6, 1] at 1: [[7, 0], [6, 1], [7, 1]]
Insert [5, 0] at 0: [[5, 0], [7, 0], [6, 1], [7, 1]]
Insert [5, 2] at 2: [[5, 0], [7, 0], [5, 2], [6, 1], [7, 1]]
Insert [4, 4] at 4: [[5, 0], [7, 0], [5, 2], [6, 1], [4, 4], [7, 1]]

Queue successfully reconstructed!
```

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        # Sort descending by height, ascending by k
        people.sort(key=lambda x: (-x[0], x[1]))
        queue = []
        
        for person in people:
            queue.insert(person[1], person)
            
        return queue
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    std::vector<std::vector<int>> reconstructQueue(std::vector<std::vector<int>>& people) {
        std::sort(people.begin(), people.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
            if (a[0] == b[0]) return a[1] < b[1];
            return a[0] > b[0];
        });

        std::vector<std::vector<int>> queue;
        for (const auto& person : people) {
            queue.insert(queue.begin() + person[1], person);
        }

        return queue;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, (a, b) -> {
            if (a[0] == b[0]) return Integer.compare(a[1], b[1]);
            return Integer.compare(b[0], a[0]);
        });

        List<int[]> queue = new LinkedList<>();
        for (int[] person : people) {
            queue.add(person[1], person);
        }

        return queue.toArray(new int[people.length][]);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N^2)$ — Sorting takes $O(N \log N)$. Each list insertion at an arbitrary index takes $O(N)$ shifting, performed $N$ times. With $N \le 2000$, $N^2 \approx 4 \times 10^6$ operations, which runs in $< 20$ ms.
- **Space Complexity:** $O(N)$ for the reconstructed list.
