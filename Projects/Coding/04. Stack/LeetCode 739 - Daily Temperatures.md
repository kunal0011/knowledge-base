---
date: "2026-09-15"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 739: Daily Temperatures"
tags:
  - leetcode
  - coding
  - stack
  - amazon
  - google
---

# LeetCode 739: Daily Temperatures

**Target Companies:** Amazon (Top #1 Monotonic Stack), Google, Meta  
**Difficulty:** Medium  
**Topic:** Monotonic Decreasing Stack

---

### Problem Statement

Given an array of integers `temperatures` represents the daily temperatures, return an array `answer` such that `answer[i]` is the number of days you have to wait after the $i$-th day to get a warmer temperature. If there is no future day for which this is possible, keep `answer[i] == 0` instead.

---

### Input & Output Formats & Constraints

- **Input:** `temperatures: List[int]`
- **Output:** `answer: List[int]` of same length
- **Constraints:**
  - $1 \le 	ext{temperatures.length} \le 10^5$
  - $30 \le 	ext{temperatures}[i] \le 100$

---

### Key Idea & Intuition

- **Naive Approach:**
  - For each day $i$, scan right to find the first $j > i$ such that `temp[j] > temp[i]`. Takes $O(N^2)$ time -> TLE.
- **Monotonic Decreasing Stack Insight:**
  - We want to find the **Next Greater Element**.
  - Maintain a stack of indices `stack = []` where the temperatures corresponding to these indices are in strictly decreasing order.
  - When we encounter day $i$ with temperature $T[i]$:
    - While the stack is not empty and $T[i] > T[	ext{stack.top()}]$:
      - We have found the warmer day for $	ext{prev\_idx} = 	ext{stack.pop()}$!
      - Set `answer[prev_idx] = i - prev_idx`.
    - Push current index $i$ onto the stack.

---

### Solution Approach (Step-by-Step)

1. Initialize `n = len(temperatures)`, `res = [0] * n`, and `stack = []`.
2. Iterate `i` from $0$ to $n - 1$:
   - While `stack` is not empty and `temperatures[i] > temperatures[stack[-1]]`:
     - `prev_idx = stack.pop()`
     - `res[prev_idx] = i - prev_idx`
   - Push `i` onto `stack`.
3. Return `res`. (Unmatched indices default to `0`).

---

### Visual Algorithm Walkthrough

```
temperatures = [73, 74, 75, 71, 69, 72, 76, 73]

i=0 (73): stack=[0]
i=1 (74): 74 > 73 -> pop 0, res[0] = 1 - 0 = 1. stack=[1]
i=2 (75): 75 > 74 -> pop 1, res[1] = 2 - 1 = 1. stack=[2]
i=3 (71): 71 < 75 -> push 3. stack=[2, 3]
i=4 (69): 69 < 71 -> push 4. stack=[2, 3, 4]
i=5 (72): 72 > 69 -> pop 4, res[4] = 5 - 4 = 1
          72 > 71 -> pop 3, res[3] = 5 - 3 = 2
          72 < 75 -> push 5. stack=[2, 5]
i=6 (76): 76 > 72 -> pop 5, res[5] = 6 - 5 = 1
          76 > 75 -> pop 2, res[2] = 6 - 2 = 4. stack=[6]
i=7 (73): 73 < 76 -> push 7. stack=[6, 7]

Final res: [1, 1, 4, 2, 1, 1, 0, 0]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Rising and Falling Sequence
- **Input:** `temperatures = [73,74,75,71,69,72,76,73]`
- **Output:** `[1,1,4,2,1,1,0,0]`

#### Example 2: Monotonically Decreasing (No Warmer Days)
- **Input:** `temperatures = [90, 80, 70]`
- **Trace:** Every element pushed, none popped.
- **Output:** `[0, 0, 0]`

#### Example 3: Strictly Increasing
- **Input:** `temperatures = [30, 40, 50, 60]`
- **Trace:** Every element immediately resolves previous day.
- **Output:** `[1, 1, 1, 0]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        res = [0] * n
        stack = []  # indices of temperatures
        
        for i, t in enumerate(temperatures):
            while stack and t > temperatures[stack[-1]]:
                prev_i = stack.pop()
                res[prev_i] = i - prev_i
            stack.append(i)
            
        return res
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <stack>

class Solution {
public:
    std::vector<int> dailyTemperatures(std::vector<int>& temperatures) {
        int n = temperatures.size();
        std::vector<int> res(n, 0);
        std::stack<int> st; // indices
        
        for (int i = 0; i < n; ++i) {
            while (!st.empty() && temperatures[i] > temperatures[st.top()]) {
                int prevIdx = st.top();
                st.pop();
                res[prevIdx] = i - prevIdx;
            }
            st.push(i);
        }
        return res;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayDeque;
import java.util.Deque;

class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] res = new int[n];
        Deque<Integer> stack = new ArrayDeque<>();
        
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                int prevIdx = stack.pop();
                res[prevIdx] = i - prevIdx;
            }
            stack.push(i);
        }
        return res;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Each index is pushed onto the stack exactly once and popped at most once.
- **Space Complexity:** $O(N)$ — In the worst-case (strictly decreasing temperatures), the stack stores all $N$ indices.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Monotonic Stack for Next Greater Element (NGE). Store *indices*, not values, to measure distance/duration.
