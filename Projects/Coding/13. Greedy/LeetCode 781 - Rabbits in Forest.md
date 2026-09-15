---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 781: Rabbits in Forest"
tags:
  - leetcode
  - coding
  - greedy
  - hash-table
  - math
  - amazon
  - google
  - bloomberg
---

# LeetCode 781: Rabbits in Forest

**Target Companies:** Amazon, Google, Bloomberg, Microsoft  
**Difficulty:** Medium  
**Topic:** Greedy / Hash Table / Math  

---

### Problem Statement

There is a forest with an unknown number of rabbits. We asked `n` rabbits "How many rabbits have the same color as you?" and collected the answers in an integer array `answers` where `answers[i]` is the answer of the $i^{\text{th}}$ rabbit.

Given the array `answers`, return *the minimum number of rabbits that could be in the forest*.

---

### Input & Output Formats & Constraints

- **Input:** An integer array `answers` ($1 \le |answers| \le 1000$).
- **Output:** An integer denoting the minimum possible number of rabbits in the forest.
- **Constraints:**
  - `1 <= answers.length <= 1000`
  - `0 <= answers[i] < 1000`

---

### Key Idea & Intuition

#### Group Size Invariant
When a rabbit says there are $x$ other rabbits with the same color, this directly implies:
$$\text{Total rabbits of this color} = x + 1$$

To minimize the total rabbit population:
1. We should greedily group as many rabbits reporting the same answer $x$ into the **same color group** as possible.
2. A single color group can accommodate at most $x + 1$ rabbits reporting answer $x$.
3. If $cnt$ rabbits report answer $x$, they cannot all be the same color once $cnt > x + 1$. They must belong to at least:
   $$\text{groups} = \left\lceil \frac{cnt}{x + 1} \right\rceil = \left\lfloor \frac{cnt + x}{x + 1} \right\rfloor$$
   distinct color groups.
4. Even if some rabbits in a group were not surveyed, that entire group of $x + 1$ rabbits must exist in the forest.
5. Therefore, the minimum number of rabbits contributed by answer $x$ is:
   $$\text{rabbits}(x) = \left\lceil \frac{cnt}{x + 1} \right\rceil \times (x + 1) = \left\lfloor \frac{cnt + x}{x + 1} \right\rfloor \times (x + 1)$$

Summing this quantity over all distinct values of $x$ gives the minimum total population.

---

### Solution Approach (Step-by-Step)

1. **Count Frequencies:**
   - Count the frequency of each distinct answer $x$ in `answers` using a hash map or frequency array.
2. **Greedy Group Calculation:**
   - Initialize `total_rabbits = 0`.
   - For each distinct pair $(x, cnt)$:
     - Group size is $g = x + 1$.
     - Number of groups is $\text{groups} = (cnt + x) / (x + 1)$ (using integer division).
     - Add $\text{groups} \times (x + 1)$ to `total_rabbits`.
3. **Return Output:**
   - Return `total_rabbits`.

---

### Visual Algorithm Walkthrough

#### Trace for `answers = [1, 1, 2]`
```
Step 1: Count answers:
  x = 1: count = 2
  x = 2: count = 1

Step 2: Process x = 1:
  - Each group can hold x + 1 = 2 rabbits.
  - We have 2 rabbits reporting 1.
  - Groups needed = ceil(2 / 2) = 1 group.
  - Population from this color = 1 group * 2 rabbits = 2 rabbits.
  [Rabbit A (white), Rabbit B (white)]

Step 3: Process x = 2:
  - Each group can hold x + 1 = 3 rabbits.
  - We have 1 rabbit reporting 2.
  - Groups needed = ceil(1 / 3) = 1 group.
  - Population from this color = 1 group * 3 rabbits = 3 rabbits.
  [Rabbit C (black), (Unseen black), (Unseen black)]

Total Minimum Rabbits = 2 + 3 = 5.
```

#### Trace for `answers = [10, 10, 10]`
```
All 3 rabbits report 10.
Group capacity = 10 + 1 = 11 rabbits.
Count = 3 <= 11 -> 1 group needed.
Total rabbits = 1 * 11 = 11.
```

---

### Solved Examples with Multiple Inputs

| Input `answers` | Frequency Map $(x \to cnt)$ | Groups per $x$: $\lceil cnt / (x+1) \rceil$ | Math Calculation | Output |
|---|---|---|---|---|
| `[1, 1, 2]` | $\{1: 2, 2: 1\}$ | $x=1 \to 1$ group; $x=2 \to 1$ group | $1 \times 2 + 1 \times 3 = 5$ | `5` |
| `[10, 10, 10]` | $\{10: 3\}$ | $x=10 \to 1$ group | $1 \times 11 = 11$ | `11` |
| `[0, 0, 1, 1, 1]` | $\{0: 2, 1: 3\}$ | $x=0 \to 2$ groups; $x=1 \to 2$ groups | $2 \times 1 + 2 \times 2 = 6$ | `6` |
| `[]` | $\emptyset$ | None | $0$ | `0` |

---

### Multi-Language Implementations

#### Python 3
```python
from collections import Counter

class Solution:
    def numRabbits(self, answers: list[int]) -> int:
        freq = Counter(answers)
        total_rabbits = 0
        
        for x, cnt in freq.items():
            group_size = x + 1
            # Number of groups needed: ceil(cnt / group_size)
            # Using integer arithmetic: (cnt + x) // group_size
            num_groups = (cnt + x) // group_size
            total_rabbits += num_groups * group_size
            
        return total_rabbits
```

#### C++17
```cpp
#include <vector>
#include <unordered_map>

class Solution {
public:
    int numRabbits(const std::vector<int>& answers) {
        std::unordered_map<int, int> freq;
        for (int ans : answers) {
            freq[ans]++;
        }
        
        int total_rabbits = 0;
        for (const auto& [x, cnt] : freq) {
            int group_size = x + 1;
            // Ceiling division: (cnt + x) / (x + 1)
            int num_groups = (cnt + x) / group_size;
            total_rabbits += num_groups * group_size;
        }
        
        return total_rabbits;
    }
};
```

#### Java 17
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public int numRabbits(int[] answers) {
        Map<Integer, Integer> freq = new HashMap<>();
        for (int ans : answers) {
            freq.put(ans, freq.getOrDefault(ans, 0) + 1);
        }
        
        int totalRabbits = 0;
        for (Map.Entry<Integer, Integer> entry : freq.entrySet()) {
            int x = entry.getKey();
            int cnt = entry.getValue();
            int groupSize = x + 1;
            
            // Integer ceiling division: (cnt + x) / (x + 1)
            int numGroups = (cnt + x) / groupSize;
            totalRabbits += numGroups * groupSize;
        }
        
        return totalRabbits;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$, where $N = |answers|$. We perform one pass to populate the frequency map and one pass over at most $N$ unique answer keys.
- **Space Complexity:** $\mathcal{O}(U)$, where $U$ is the number of distinct answers in `answers`. Since $0 \le answers[i] < 1000$, $U \le 1000$, which is $\mathcal{O}(1)$ auxiliary space in practice.

---

### Takeaway Pattern & Interview Traps

1. **Integer Ceiling Arithmetic:** $\lceil a / b \rceil = (a + b - 1) / b$. Since $b = x + 1$, $(cnt + (x + 1) - 1) / (x + 1) = (cnt + x) / (x + 1)$. This avoids floating-point precision issues in C++ and Java.
2. **Unseen Rabbits:** A common error is summing only surveyed rabbits. If 1 rabbit reports 2, there must be at least 3 rabbits of that color in the forest, even though the other 2 were never interviewed.
3. **Zero Answer Case ($x = 0$):** When $x = 0$, each rabbit is unique in color; group size is 1, and each contributes $1 \times 1 = 1$, correctly giving $cnt$ total rabbits.