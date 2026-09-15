---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 135: Candy"
tags:
  - leetcode
  - coding
  - greedy
  - amazon
  - google
---

# LeetCode 135: Candy

**Target Companies:** Google (Signature Greedy Hard), Amazon, Meta  
**Difficulty:** Hard  
**Topic:** Two-Pass Greedy Slope Invariant

---

### Problem Statement

There are $n$ children standing in a line. Each child is assigned a rating value given in the integer array `ratings`.

You are giving candies to these children subjected to the following requirements:
1. Each child must have at least one candy.
2. Children with a higher rating get more candies than their neighbors.

Return the **minimum number of candies** you need to have to distribute the candies to the children.

---

### Input & Output Formats & Constraints

- **Input:** `ratings: List[int]`
- **Output:** `int` (total sum of candies)
- **Constraints:**
  - $n == \text{ratings.length}$
  - $1 \le n \le 2 \times 10^4$
  - $0 \le \text{ratings}[i] \le 2 \times 10^4$

---

### Key Idea & Intuition

- **Decomposing Neighbor Constraints:**
  - Requirement 2 says:
    - If `ratings[i] > ratings[i - 1]`: `candies[i] > candies[i - 1]`.
    - If `ratings[i] > ratings[i + 1]`: `candies[i] > candies[i + 1]`.
  - Trying to satisfy both left and right neighbors simultaneously in a single pass is tricky.
- **Two-Pass Solution:**
  1. **Left-to-Right Pass:**
     - Initialize every child with 1 candy.
     - For $i$ from 1 to $n - 1$: if `ratings[i] > ratings[i - 1]`, set `candies[i] = candies[i - 1] + 1`.
  2. **Right-to-Left Pass:**
     - For $i$ from $n - 2$ down to 0: if `ratings[i] > ratings[i + 1]`, set `candies[i] = max(candies[i], candies[i + 1] + 1)`.
  - Taking `max` ensures that both the left-neighbor condition and right-neighbor condition remain satisfied!

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        candies = [1] * n
        
        # Left to right
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1
                
        # Right to left
        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                candies[i] = max(candies[i], candies[i + 1] + 1)
                
        return sum(candies)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <numeric>
#include <algorithm>

class Solution {
public:
    int candy(std::vector<int>& ratings) {
        int n = ratings.size();
        std::vector<int> candies(n, 1);

        for (int i = 1; i < n; ++i) {
            if (ratings[i] > ratings[i - 1]) {
                candies[i] = candies[i - 1] + 1;
            }
        }

        for (int i = n - 2; i >= 0; --i) {
            if (ratings[i] > ratings[i + 1]) {
                candies[i] = std::max(candies[i], candies[i + 1] + 1);
            }
        }

        return std::accumulate(candies.begin(), candies.end(), 0);
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public int candy(int[] ratings) {
        int n = ratings.length;
        int[] candies = new int[n];
        Arrays.fill(candies, 1);

        for (int i = 1; i < n; i++) {
            if (ratings[i] > ratings[i - 1]) {
                candies[i] = candies[i - 1] + 1;
            }
        }

        for (int i = n - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1]) {
                candies[i] = Math.max(candies[i], candies[i + 1] + 1);
            }
        }

        int total = 0;
        for (int c : candies) total += c;
        return total;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Two linear scans across the array.
- **Space Complexity:** $O(N)$ for the candy distribution array.
