---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 486: Predict the Winner"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - interval-dp
  - minimax
  - game-theory
  - google
---

# LeetCode 486: Predict the Winner

**Target Companies:** Google, Amazon, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Interval Dynamic Programming / Minimax Game Theory / Relative Score Difference  

---

### Problem Statement

You are given an integer array `nums`. Two players are playing a game with this array: player 1 and player 2.

Player 1 and player 2 take turns, with player 1 starting first. Both players start the game with a score of `0`. At each turn, the player takes one of the numbers from either end of the array (i.e., `nums[0]` or `nums[nums.length - 1]`), which reduces the size of the array by `1`. The player adds the chosen number to their score. The game ends when there are no more elements in the array.

Return `true` if Player 1 can win the game. If the scores of both players are equal, player 1 is still considered the winner, and you should also return `true`. You may assume that both players are playing optimally.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]` — Array of integers representing score values.
- **Output:** `bool` — `true` if Player 1 can guarantee $\text{Score}_1 \ge \text{Score}_2$, `false` otherwise.
- **Constraints:**
  - $1 \le \text{nums.length} \le 20$
  - $0 \le \text{nums}[i] \le 10^7$

---

### Key Idea & Intuition

1. **Relative Score Advantage (Minimax):**
   - In a zero-sum, two-player perfect information game, tracking absolute scores for both players creates unnecessary 3D states.
   - Instead, track the **net relative score difference**:
     $$\text{net score} = (\text{current player's score}) - (\text{opponent's score})$$
   - If Player 1 can ensure a net relative score $\ge 0$ at the end, Player 1 wins or ties.

2. **Interval DP Formulation:**
   - Let $\text{dp}[i][j]$ be the maximum relative score difference the current player can achieve from the subarray $\text{nums}[i \dots j]$.
   - At state $[i, j]$, the current player has two choices:
     1. **Pick the left element $\text{nums}[i]$:**
        - Gaining $\text{nums}[i]$.
        - The opponent then plays optimally on the remaining subsegment $[i + 1, j]$, achieving a relative score advantage of $\text{dp}[i + 1][j]$.
        - Net gain from this choice: $\text{nums}[i] - \text{dp}[i + 1][j]$.
     2. **Pick the right element $\text{nums}[j]$:**
        - Gaining $\text{nums}[j]$.
        - The opponent achieves a relative advantage of $\text{dp}[i][j - 1]$ on $[i, j - 1]$.
        - Net gain from this choice: $\text{nums}[j] - \text{dp}[i][j - 1]$.
   - The current player maximizes their net advantage:
     $$\text{dp}[i][j] = \max\left( \text{nums}[i] - \text{dp}[i + 1][j], \ \text{nums}[j] - \text{dp}[i][j - 1] \right)$$

3. **Base Case:**
   - Subarray of length 1: $\text{dp}[i][i] = \text{nums}[i]$ (the player takes the only available number; the opponent receives 0).

4. **1D Space Optimization:**
   - Notice that $\text{dp}[i][j]$ depends only on $\text{dp}[i + 1][j]$ (from the same subproblem length or previous $i$) and $\text{dp}[i][j - 1]$.
   - We can compress the 2D table into a 1D array of size $n$, updating backwards.

---

### Solution Approach (Step-by-Step)

1. **Table Initialization:**
   - Let $n = \text{len}(nums)$.
   - Initialize 1D array `dp` where `dp[i] = nums[i]` for $i \in [0, n - 1]$.
2. **Bottom-Up Interval Traversal:**
   - Iterate $i$ backwards from $n - 2$ down to $0$:
     - For $j$ from $i + 1$ to $n - 1$:
       - `dp[j] = max(nums[i] - dp[j], nums[j] - dp[j - 1])`
3. **Evaluate Result:**
   - Return `dp[n - 1] >= 0`.

---

### Visual Algorithm Walkthrough

For `nums = [1, 5, 2]`:

```
Base Cases (Length 1):
  dp[0][0] = 1, dp[1][1] = 5, dp[2][2] = 2

Length 2:
  dp[0][1] (nums=[1, 5]):
    Pick 1: 1 - dp[1][1] = 1 - 5 = -4
    Pick 5: 5 - dp[0][0] = 5 - 1 = 4
    dp[0][1] = max(-4, 4) = 4

  dp[1][2] (nums=[5, 2]):
    Pick 5: 5 - dp[2][2] = 5 - 2 = 3
    Pick 2: 2 - dp[1][1] = 2 - 5 = -3
    dp[1][2] = max(3, -3) = 3

Length 3 (nums=[1, 5, 2]):
  dp[0][2]:
    Pick 1: 1 - dp[1][2] = 1 - 3 = -2
    Pick 2: 2 - dp[0][1] = 2 - 4 = -2
    dp[0][2] = max(-2, -2) = -2

Final Value: dp[0][2] = -2.
Since -2 < 0, Player 1 cannot win or tie.
Result: False!
```

---

### Solved Examples with Multiple Inputs

| Case | `nums` | Relative Difference $dp[0][n-1]$ | Result | Explanation |
|---|---|---|---|---|
| **P1 Loses** | `[1, 5, 2]` | `-2` | `false` | Whatever P1 takes, P2 grabs `5` and wins |
| **P1 Wins** | `[1, 5, 233, 7]` | `222` | `true` | P1 can secure the high-value `233` |
| **Even Length** | `[1, 10, 100, 10]` | `99` | `true` | Even length always provides parity advantage |
| **Single Element** | `[10]` | `10` | `true` | P1 takes the only element |
| **Two Elements** | `[3, 7]` | `4` | `true` | P1 picks larger element `7` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def predictTheWinner(self, nums: List[int]) -> bool:
        n = len(nums)
        # Even length array is an automatic win for Player 1 (parity strategy)
        if n % 2 == 0:
            return True
            
        # dp[j] stores max relative score advantage for subarray nums[i...j]
        dp = nums[:]
        
        for i in range(n - 2, -1, -1):
            for j in range(i + 1, n):
                dp[j] = max(nums[i] - dp[j], nums[j] - dp[j - 1])
                
        return dp[n - 1] >= 0
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    bool predictTheWinner(std::vector<int>& nums) {
        int n = nums.size();
        if (n % 2 == 0) return true;

        std::vector<int> dp = nums;

        for (int i = n - 2; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                dp[j] = std::max(nums[i] - dp[j], nums[j] - dp[j - 1]);
            }
        }

        return dp[n - 1] >= 0;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public boolean predictTheWinner(int[] nums) {
        int n = nums.length;
        if (n % 2 == 0) return true;

        int[] dp = nums.clone();

        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                dp[j] = Math.max(nums[i] - dp[j], nums[j] - dp[j - 1]);
            }
        }

        return dp[n - 1] >= 0;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N^2)$  
  There are $\frac{N(N - 1)}{2}$ sub-intervals evaluated. With $N \le 20$, the number of operations is $\le 200$, executing in $< 0.1$ ms.
- **Space Complexity:** $\mathcal{O}(N)$  
  Using 1D rolling array space optimization, only an array of size $N \le 20$ is maintained.

---

### Takeaway Pattern & Interview Traps

1. **Even Length Parity Trick (Stone Game):**
   - When $n$ is even, Player 1 can always choose to take all even-indexed elements or all odd-indexed elements. Because Player 1 can compare $\sum \text{even}$ vs $\sum \text{odd}$ before making the first move, Player 1 can force a win whenever $n$ is even (as seen in LeetCode 877: Stone Game)!
2. **Minimax through Subtraction:**
   - Notice how `nums[i] - dp[next]` elegantly encapsulates both maximizing current player's score and minimizing opponent's score without needing alternating player flags or mutual recursion.