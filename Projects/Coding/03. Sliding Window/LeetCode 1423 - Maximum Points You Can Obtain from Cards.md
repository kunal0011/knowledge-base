---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1423: Maximum Points You Can Obtain from Cards"
tags:
  - leetcode
  - coding
  - sliding-window
  - array
  - prefix-sum
  - amazon
  - google
---

# LeetCode 1423: Maximum Points You Can Obtain from Cards

**Target Companies:** Google (Signature Favorite), Amazon, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Fixed-Size Sliding Window / Inverted Problem Formulation  

---

### Problem Statement

There are several cards arranged in a row, and each card has an associated number of points. The points are given in the integer array `cardPoints`.

In one step, you can take one card from the beginning or from the end of the row. You have to take exactly `k` cards.

Your score is the sum of the points of the cards you have taken.

Given the integer array `cardPoints` and the integer `k`, return *the maximum score you can obtain*.

---

### Input & Output Formats & Constraints

- **Input:** `cardPoints: List[int]`, `k: int`
- **Output:** `int` (maximum sum of $k$ cards taken from the ends)
- **Constraints:**
  - $1 \le \text{cardPoints.length} \le 10^5$
  - $1 \le \text{cardPoints}[i] \le 10^4$
  - $1 \le k \le \text{cardPoints.length}$

---

### Key Idea & Intuition

- **The Complementary Formulation (Inverting the Problem):**
  - Taking $k$ cards from the two ends (left and right) leaves behind a **contiguous remaining subarray of length $W = N - k$** in the middle.
  - The total sum of all cards in the array is fixed:
    $$\text{Total} = \sum_{i=0}^{N-1} \text{cardPoints}[i]$$
  - Therefore, maximizing the sum of the $k$ chosen cards from the ends is strictly equivalent to:
    $$\textbf{Minimizing the sum of a contiguous subarray of fixed length } (N - k)$$
  - Once we find the minimum sum of any subarray of length $N - k$, the answer is simply:
    $$\max(\text{score}) = \text{Total} - \min(\text{subarray sum of length } N - k)$$
- **Special Case ($k == N$):**
  - If $k == N$, we take all cards $\implies$ return `sum(cardPoints)`.

---

### Solution Approach (Step-by-Step)

1. Let $N = \text{len}(cardPoints)$ and $W = N - k$.
2. Compute `total_sum = sum(cardPoints)`.
3. If $W == 0$: return `total_sum`.
4. Compute the sum of the first window of size $W$: `curr_sum = sum(cardPoints[:W])`.
5. Initialize `min_window_sum = curr_sum`.
6. Slide the window of length $W$ from index $W$ to $N - 1$:
   - `curr_sum += cardPoints[i] - cardPoints[i - W]`
   - `min_window_sum = min(min_window_sum, curr_sum)`
7. Return `total_sum - min_window_sum`.

---

### Visual Algorithm Walkthrough

Let `cardPoints = [1, 2, 3, 4, 5, 6, 1]`, $k = 3$.
- Array length $N = 7$.
- Window size to leave in the middle $W = N - k = 7 - 3 = 4$.
- Total sum $= 1 + 2 + 3 + 4 + 5 + 6 + 1 = 22$.

```
Indices:      0   1   2   3   4   5   6
Cards:       [1,  2,  3,  4,  5,  6,  1]

Slide window of size W = 4:
Window 0..3: [1,  2,  3,  4]           -> sum = 10 -> score = 22 - 10 = 12
Window 1..4:     [2,  3,  4,  5]       -> sum = 14 -> score = 22 - 14 = 8
Window 2..5:         [3,  4,  5,  6]   -> sum = 18 -> score = 22 - 18 = 4
Window 3..6:             [4,  5,  6,  1] -> sum = 16 -> score = 22 - 16 = 6

Minimum middle window sum = 10 (indices 0..3).
Complementary chosen cards: indices 4, 5, 6 -> [5, 6, 1].
Maximum score = 22 - 10 = 12 (or 5 + 6 + 1 = 12)!
```

---

### Solved Examples with Multiple Inputs

| Test Case | `cardPoints` | `k` | Window $N - k$ | Min Window Sum | Max Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | `[1, 2, 3, 4, 5, 6, 1]` | `3` | 4 | 10 (`[1, 2, 3, 4]`) | `12` |
| **Pick All** | `[2, 2, 2]` | `3` | 0 | 0 | `6` |
| **Split Ends** | `[9, 7, 7, 9, 7, 7, 9]` | `7` | 0 | 0 | `55` |
| **Take 1 Card** | `[1, 1000, 1]` | `1` | 2 | 1001 (`[1, 1000]` or `[1000, 1]`) | `1` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        """
        Finds the maximum points obtainable by taking k cards from the ends.
        Inverts the problem to find the minimum sum subarray of size N - k.
        """
        n = len(cardPoints)
        window_size = n - k
        total_sum = sum(cardPoints)

        if window_size == 0:
            return total_sum

        # Calculate initial window of size N - k
        curr_sum = sum(cardPoints[:window_size])
        min_window_sum = curr_sum

        # Slide window across the array
        for i in range(window_size, n):
            curr_sum += cardPoints[i] - cardPoints[i - window_size]
            if curr_sum < min_window_sum:
                min_window_sum = curr_sum

        return total_sum - min_window_sum
```

#### C++17
```cpp
#include <vector>
#include <numeric>
#include <algorithm>

class Solution {
public:
    int maxScore(const std::vector<int>& cardPoints, int k) {
        int n = static_cast<int>(cardPoints.size());
        int window_size = n - k;
        int total_sum = std::accumulate(cardPoints.begin(), cardPoints.end(), 0);

        if (window_size == 0) {
            return total_sum;
        }

        int curr_sum = 0;
        for (int i = 0; i < window_size; ++i) {
            curr_sum += cardPoints[i];
        }

        int min_window_sum = curr_sum;
        for (int i = window_size; i < n; ++i) {
            curr_sum += cardPoints[i] - cardPoints[i - window_size];
            min_window_sum = std::min(min_window_sum, curr_sum);
        }

        return total_sum - min_window_sum;
    }
};
```

#### Java
```java
class Solution {
    public int maxScore(int[] cardPoints, int k) {
        int n = cardPoints.length;
        int windowSize = n - k;
        int totalSum = 0;

        for (int point : cardPoints) {
            totalSum += point;
        }

        if (windowSize == 0) {
            return totalSum;
        }

        int currSum = 0;
        for (int i = 0; i < windowSize; i++) {
            currSum += cardPoints[i];
        }

        int minWindowSum = currSum;
        for (int i = windowSize; i < n; i++) {
            currSum += cardPoints[i] - cardPoints[i - windowSize];
            minWindowSum = Math.min(minWindowSum, currSum);
        }

        return totalSum - minWindowSum;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ where $N = \text{cardPoints.length}$.
  - Computing `total_sum` takes $\mathcal{O}(N)$.
  - Sliding the window of size $N - k$ across the array takes $\mathcal{O}(N)$ constant-time additions and subtractions.
  - Overall time is strictly linear, running in $< 2 \text{ ms}$ for $N = 10^5$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space since only a few scalar accumulator variables are maintained.

---

### Takeaway Pattern & Interview Traps

- **The Complementary Subarray Pattern:** When a problem asks to take $k$ elements from the boundaries (head and tail), always ask: *What remains in the middle?* The middle is always a contiguous subarray of size $N - k$, turning a disjoint-ends problem into a standard contiguous sliding window.
- **Edge Case $k == N$:** When $k == N$, the middle window size is $0$. Handle this boundary case gracefully without index-out-of-bounds.