---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 506: Relative Ranks"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
  - sorting
  - amazon
  - google
---

# LeetCode 506: Relative Ranks

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Easy  
**Topic:** Priority Queue (Max-Heap) / Sorting with Index Tracking

---

### Problem Statement

You are given an integer array `score` of size `n`, where `score[i]` is the score of the $i^{\text{th}}$ athlete in a competition. All the scores are guaranteed to be **unique**.

The athletes are placed based on their scores, where the $1^{\text{st}}$ place athlete has the highest score, the $2^{\text{nd}}$ place athlete has the $2^{\text{nd}}$ highest score, and so on. The placement of each athlete determines their rank:
- The $1^{\text{st}}$ place athlete's rank is `"Gold Medal"`.
- The $2^{\text{nd}}$ place athlete's rank is `"Silver Medal"`.
- The $3^{\text{rd}}$ place athlete's rank is `"Bronze Medal"`.
- For the $4^{\text{th}}$ place to the $n^{\text{th}}$ place athlete, their rank is their placement number (i.e., the $x^{\text{th}}$ place athlete's rank is `str(x)`).

Return an array `answer` of size `n` where `answer[i]` is the rank of the $i^{\text{th}}$ athlete.

---

### Input & Output Formats & Constraints

- **Input:**
  - `score`: `List[int]`, where $n = \text{len}(score)$. $1 \le n \le 10^4$.
  - $0 \le score[i] \le 10^6$.
  - All values in `score` are distinct.
- **Output:**
  - `List[str]`: Array of strings where each element corresponds to the rank of the original athlete at index $i$.
- **Constraints:**
  - Distinct scores.
  - Ranks $1$, $2$, $3$ require verbatim medal names.

---

### Key Idea & Intuition

The problem requires ordering athletes by descending score while returning their assigned ranks according to their **original input indices**.

#### Priority Queue (Max-Heap) Approach:
1. Build a max-heap of pairs: `(-score[i], i)`.
2. Extract the highest score one by one:
   - Rank 1 $\implies$ assign `"Gold Medal"` to `answer[i]`.
   - Rank 2 $\implies$ assign `"Silver Medal"` to `answer[i]`.
   - Rank 3 $\implies$ assign `"Bronze Medal"` to `answer[i]`.
   - Rank $r \ge 4 \implies$ assign `str(r)` to `answer[i]`.
3. This pattern directly matches the fundamental usage of priority queues for continuous top-k extraction.

#### Alternative Sorting Indices Approach:
- Sort a list of indices $[0 \dots n-1]$ based on $score[idx]$ descending. Both approaches achieve $\mathcal{O}(N \log N)$ time.

---

### Solution Approach (Step-by-Step)

1. Initialize `n = len(score)` and `result = [""] * n`.
2. Push all `(-score[i], i)` into a list `max_heap` and call `heapify`.
3. Initialize `rank = 1`.
4. While `max_heap` is non-empty:
   - `_, idx = heappop(max_heap)`.
   - If `rank == 1`: `result[idx] = "Gold Medal"`.
   - Else if `rank == 2`: `result[idx] = "Silver Medal"`.
   - Else if `rank == 3`: `result[idx] = "Bronze Medal"`.
   - Else: `result[idx] = str(rank)`.
   - Increment `rank += 1`.
5. Return `result`.

---

### Visual Algorithm Walkthrough

Let `score = [10, 3, 8, 9, 4]`:

```
Input indices and scores:
  Index 0: 10
  Index 1: 3
  Index 2: 8
  Index 3: 9
  Index 4: 4

Max-Heap elements (-score, index):
  [ (-10, 0), (-9, 3), (-8, 2), (-4, 4), (-3, 1) ]

Extraction Sequence:
  Rank 1: Pop (-10, index 0) -> result[0] = "Gold Medal"
  Rank 2: Pop (-9,  index 3) -> result[3] = "Silver Medal"
  Rank 3: Pop (-8,  index 2) -> result[2] = "Bronze Medal"
  Rank 4: Pop (-4,  index 4) -> result[4] = "4"
  Rank 5: Pop (-3,  index 1) -> result[1] = "5"

Final result array:
  ["Gold Medal", "5", "Bronze Medal", "Silver Medal", "4"]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Scores

- **Input:** `score = [5, 4, 3, 2, 1]`
- **Output:** `["Gold Medal", "Silver Medal", "Bronze Medal", "4", "5"]`

#### Example 2: Scrambled Placement

- **Input:** `score = [10, 3, 8, 9, 4]`
- **Output:** `["Gold Medal", "5", "Bronze Medal", "Silver Medal", "4"]`

#### Example 3: Small Arrays ($N \le 2$)

- **Input:** `score = [1]` $\implies$ `["Gold Medal"]`
- **Input:** `score = [2, 10]` $\implies$ `["Silver Medal", "Gold Medal"]`

---

### Multi-Language Implementations

#### Python 3

```python
import heapq
from typing import List

class Solution:
    def findRelativeRanks(self, score: List[int]) -> List[str]:
        n = len(score)
        result: List[str] = [""] * n

        # Max-heap storing (-score, original_index)
        max_heap = [(-s, i) for i, s in enumerate(score)]
        heapq.heapify(max_heap)

        rank = 1
        while max_heap:
            _, idx = heapq.heappop(max_heap)

            if rank == 1:
                result[idx] = "Gold Medal"
            elif rank == 2:
                result[idx] = "Silver Medal"
            elif rank == 3:
                result[idx] = "Bronze Medal"
            else:
                result[idx] = str(rank)

            rank += 1

        return result
```

#### C++17

```cpp
#include <vector>
#include <string>
#include <queue>

class Solution {
public:
    std::vector<std::string> findRelativeRanks(const std::vector<int>& score) {
        int n = static_cast<int>(score.size());
        std::vector<std::string> result(n);

        // Max-heap storing pair of (score, original_index)
        std::priority_queue<std::pair<int, int>> max_heap;
        for (int i = 0; i < n; ++i) {
            max_heap.emplace(score[i], i);
        }

        int rank = 1;
        while (!max_heap.empty()) {
            auto [s, idx] = max_heap.top();
            max_heap.pop();

            if (rank == 1) {
                result[idx] = "Gold Medal";
            } else if (rank == 2) {
                result[idx] = "Silver Medal";
            } else if (rank == 3) {
                result[idx] = "Bronze Medal";
            } else {
                result[idx] = std::to_string(rank);
            }

            rank++;
        }

        return result;
    }
};
```

#### Java

```java
import java.util.PriorityQueue;

public class Solution {
    public String[] findRelativeRanks(int[] score) {
        int n = score.length;
        String[] result = new String[n];

        // Max-heap storing [score, index]
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>(
            (a, b) -> Integer.compare(b[0], a[0])
        );

        for (int i = 0; i < n; i++) {
            maxHeap.offer(new int[]{score[i], i});
        }

        int rank = 1;
        while (!maxHeap.isEmpty()) {
            int[] top = maxHeap.poll();
            int idx = top[1];

            if (rank == 1) {
                result[idx] = "Gold Medal";
            } else if (rank == 2) {
                result[idx] = "Silver Medal";
            } else if (rank == 3) {
                result[idx] = "Bronze Medal";
            } else {
                result[idx] = String.valueOf(rank);
            }

            rank++;
        }

        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \log N)$
  - Building the heap from $N$ elements takes $\mathcal{O}(N)$ (via `heapify`).
  - Popping all $N$ elements takes $N \times \mathcal{O}(\log N) = \mathcal{O}(N \log N)$ time.
  - Total Time: $\mathcal{O}(N \log N)$.
- **Space Complexity:** $\mathcal{O}(N)$
  - The heap and result array store $N$ elements.

---

### Takeaway Pattern & Interview Traps

1. **Index Retention:**
   - Always retain the original index of each item in the heap tuple `(score, index)` so that ranks can be written directly to the target output slots without a secondary lookup.
2. **Medal Naming Exactness:**
   - Watch out for exact casing and spacing: `"Gold Medal"`, `"Silver Medal"`, `"Bronze Medal"`.