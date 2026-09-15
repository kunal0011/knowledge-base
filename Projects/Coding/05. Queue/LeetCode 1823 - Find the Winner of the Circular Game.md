---
date: "2026-09-15"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 1823: Find the Winner of the Circular Game"
tags:
  - leetcode
  - coding
  - queue
  - josephus-problem
  - recursion
  - math
  - google
  - amazon
---

# LeetCode 1823: Find the Winner of the Circular Game

**Target Companies:** Google, Amazon, Bloomberg, Microsoft  
**Difficulty:** Medium  
**Topic:** The Josephus Problem / Queue Simulation & $O(N)$ Iterative Recurrence  

---

### Problem Statement

There are $n$ friends that are playing a game. The friends are sitting in a circle and are numbered from $1$ to $n$ in **clockwise order**. More formally, moving clockwise from the $i$-th friend brings you to the $(i+1)$-th friend for $1 \le i < n$, and moving clockwise from the $n$-th friend brings you to the $1$-st friend.

The rules of the game are as follows:
1. Start at the $1$-st friend.
2. Count the next $k$ friends in the clockwise direction including the friend you started at. The counting wraps around the circle and may count some friends more than once.
3. The last friend you counted leaves the circle and loses the game.
4. If there is still more than one friend in the circle, go back to step 2 starting from the friend **immediately clockwise** of the friend who just lost and repeat.
5. Else, the last friend in the circle wins the game.

Given the number of friends, $n$, and an integer $k$, return the **winner of the game**.

---

### Input & Output Formats & Constraints

- **Input:** `n: int`, `k: int`
- **Output:** `int` (1-indexed friend number)
- **Constraints:**
  - $1 \le k \le n \le 500$

---

### Key Idea & Intuition

This is the historic **Josephus Problem**. There are two standard approaches:

#### Approach 1: FIFO Queue Simulation ($O(N \cdot k)$ Time, $O(N)$ Space)
A queue directly mimics circular elimination:
- Insert all numbers $1 \dots n$ into a queue.
- For each elimination round:
  - Rotate the front element to the back $k - 1$ times (`q.append(q.popleft())`).
  - Pop the front element (`q.popleft()`), permanently eliminating the $k$-th friend.
- Repeat until 1 element remains.

#### Approach 2: Optimal Mathematical Recurrence ($O(N)$ Time, $O(1)$ Space)
Consider the game with 0-indexed positions:
- When 1 person remains, their index in a 1-person circle is $0$.
- In a circle of size $i$, if we know the safe position in a circle of size $i - 1$, shifting back to size $i$ simply shifts indices forward by $k$:
$$J(1) = 0$$
$$J(i) = (J(i - 1) + k) \pmod i \quad \text{for } i \in [2, n]$$
- Finally, convert the 0-indexed result to 1-indexed: $\text{Winner} = J(n) + 1$.

---

### Solution Approach (Step-by-Step)

#### Optimal Mathematical Formulation:
1. Initialize `ans = 0` (0-indexed base case for 1 friend).
2. Loop `i` from $2$ to $n$:
   - `ans = (ans + k) % i`.
3. Return `ans + 1` (converting to 1-indexed).

---

### Visual Algorithm Walkthrough

```
n = 5 friends: [1, 2, 3, 4, 5], k = 2

Round 1:
  Circle: (1) -> (2) -> (3) -> (4) -> (5) -> (1)
  Start at 1, count 2: 1, 2. Friend 2 eliminated!
  Next start: 3. Remaining: [1, 3, 4, 5]

Round 2:
  Start at 3, count 2: 3, 4. Friend 4 eliminated!
  Next start: 5. Remaining: [1, 3, 5]

Round 3:
  Start at 5, count 2: 5, 1 (wraps!). Friend 1 eliminated!
  Next start: 3. Remaining: [3, 5]

Round 4:
  Start at 3, count 2: 3, 5. Friend 5 eliminated!
  Remaining: [3]

Winner: Friend 3!

Mathematical Recurrence Check (0-indexed):
  i = 1: J(1) = 0
  i = 2: J(2) = (0 + 2) % 2 = 0
  i = 3: J(3) = (0 + 2) % 3 = 2
  i = 4: J(4) = (2 + 2) % 4 = 0
  i = 5: J(5) = (0 + 2) % 5 = 2
1-indexed winner: J(5) + 1 = 2 + 1 = 3! Exactly matches!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Step
- **Input:** `n = 5`, `k = 2`
- **Output:** `3`

#### Example 2: Larger Step Than Size
- **Input:** `n = 6`, `k = 5`
- **Eliminations:** 5, 4, 6, 2, 3 $\to$ Winner is 1.
- **Output:** `1`

#### Example 3: Trivial Single Player ($n = 1$)
- **Input:** `n = 1`, `k = 1`
- **Output:** `1`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        # Optimal O(N) Time, O(1) Space Josephus Recurrence
        winner = 0  # J(1) = 0 (0-indexed)
        for i in range(2, n + 1):
            winner = (winner + k) % i
            
        return winner + 1
```

#### 2. C++ (C++17 / STL)
```cpp
class Solution {
public:
    int findTheWinner(int n, int k) {
        // Optimal O(N) Time, O(1) Space
        int winner = 0; // 0-indexed
        for (int i = 2; i <= n; ++i) {
            winner = (winner + k) % i;
        }
        return winner + 1;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int findTheWinner(int n, int k) {
        // Optimal O(N) Time, O(1) Space
        int winner = 0;
        for (int i = 2; i <= n; i++) {
            winner = (winner + k) % i;
        }
        return winner + 1;
    }
}
```

#### 4. Queue Simulation Alternative (Python 3)
```python
from collections import deque

class SolutionQueueSimulation:
    def findTheWinner(self, n: int, k: int) -> int:
        q = deque(range(1, n + 1))
        while len(q) > 1:
            for _ in range(k - 1):
                q.append(q.popleft())
            q.popleft()  # Eliminate k-th friend
        return q[0]
```

---

### Complexity Analysis

- **Time Complexity:**
  - **Mathematical Approach:** $O(N)$ — Single loop from $2$ to $n$.
  - **Queue Simulation:** $O(N \cdot k)$ — $N - 1$ rounds with $k$ rotations per round.
- **Space Complexity:**
  - **Mathematical Approach:** $O(1)$ auxiliary space.
  - **Queue Simulation:** $O(N)$ auxiliary space for the deque.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** The Josephus Problem $J(n, k) = (J(n - 1, k) + k) \pmod n$.
- **Trap:** Forgetting that modulo arithmetic only works cleanly on **0-indexed** systems ($0 \dots n - 1$). If you try to do `(val + k) % n` directly on 1-indexed numbers, when `val + k == n`, modulo produces `0` instead of `n`. Always convert to 0-indexed first, then add 1 at the end.