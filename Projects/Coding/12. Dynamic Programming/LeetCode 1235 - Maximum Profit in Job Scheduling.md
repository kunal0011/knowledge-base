---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 1235: Maximum Profit in Job Scheduling"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - binary-search
  - amazon
  - google
---

# LeetCode 1235: Maximum Profit in Job Scheduling

**Target Companies:** Google (Signature DP Hard), Amazon (Top #1), Meta  
**Difficulty:** Hard  
**Topic:** DP + Binary Search (Weighted Interval Scheduling)

---

### Problem Statement

We have $n$ jobs, where every job is scheduled to be done from `startTime[i]` to `endTime[i]`, obtaining a profit of `profit[i]`.

You're given the `startTime`, `endTime` and `profit` arrays, return the maximum profit you can take such that there are no two jobs in the subset with overlapping time range.

If you choose a job that ends at time `X` you will be able to start another job that starts at time `X`.

---

### Input & Output Formats & Constraints

- **Input:** `startTime: List[int]`, `endTime: List[int]`, `profit: List[int]`
- **Output:** `int` (maximum total profit)
- **Constraints:**
  - $1 \le \text{startTime.length} == \text{endTime.length} == \text{profit.length} \le 5 \times 10^4$
  - $1 \le \text{startTime}[i] < \text{endTime}[i] \le 10^9$
  - $1 \le \text{profit}[i] \le 10^4$

---

### Key Idea & Intuition

- **Weighted Interval Scheduling Formulation:**
  - Classic greedy interval scheduling (choosing earliest end time) fails because jobs have unequal profits.
- **Dynamic Programming Choice:**
  - Sort all jobs by their `endTime`.
  - For job $i$ with `(start, end, profit)`:
    - **Option 1 (Skip job $i$):** Take the maximum profit achieved up to job $i - 1$: `dp[i - 1]`.
    - **Option 2 (Take job $i$):** Earn `profit` plus the maximum profit from the latest non-overlapping job $j$ that ends $\le start_i$.
  - $$\text{DP}[i] = \max(\text{DP}[i - 1], \text{profit}_i + \text{DP}[j])$$
  - Finding the latest non-overlapping job $j$ efficiently:
    - Because jobs are sorted by `endTime`, we can find $j$ in **$O(\log N)$** time using **Binary Search** (`bisect_right`).

---

### Solution Approach (Step-by-Step)

1. Combine jobs into tuples `(startTime, endTime, profit)` and sort by `endTime`:
   - `jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])`
2. Extract sorted `end_times = [job[1] for job in jobs]`.
3. Initialize `dp = [0] * (n + 1)`:
   - `dp[k]` stores the max profit considering the first $k$ jobs.
4. For $i$ from $1$ to $n$:
   - `start, end, profit = jobs[i - 1]`
   - Find latest compatible job index $j$ using `bisect_right(end_times, start)`:
     - `j = bisect_right(end_times, start, 0, i - 1)`
   - `dp[i] = max(dp[i - 1], profit + dp[j])`
5. Return `dp[n]`.

---

### Visual Algorithm Walkthrough

```
Jobs sorted by endTime:
Job 0: [1, 3], profit = 50
Job 1: [2, 4], profit = 10
Job 2: [3, 5], profit = 40
Job 3: [3, 6], profit = 70

DP Array Evolution:
dp[0] = 0

i=1: Job 0 [1,3], profit 50.
     Compatible job: none (j=0) -> dp[1] = max(dp[0], 50 + dp[0]) = 50

i=2: Job 1 [2,4], profit 10.
     Compatible job: none ending <= 2 -> dp[2] = max(dp[1], 10 + 0) = 50

i=3: Job 2 [3,5], profit 40.
     Compatible job: Job 0 ends at 3 <= 3 (j=1) -> dp[3] = max(dp[2], 40 + dp[1]) = max(50, 40 + 50) = 90

i=4: Job 3 [3,6], profit 70.
     Compatible job: Job 0 ends at 3 <= 3 (j=1) -> dp[4] = max(dp[3], 70 + dp[1]) = max(90, 70 + 50) = 120

Max Profit: 120 (Jobs 0 + 3)
```

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from bisect import bisect_right

class Solution:
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])
        n = len(jobs)
        end_times = [j[1] for j in jobs]
        
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            s, e, p = jobs[i - 1]
            # Find latest job ending <= s
            prev_idx = bisect_right(end_times, s, 0, i - 1)
            dp[i] = max(dp[i - 1], p + dp[prev_idx])
            
        return dp[n]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

struct Job {
    int start, end, profit;
};

class Solution {
public:
    int jobScheduling(std::vector<int>& startTime, std::vector<int>& endTime, std::vector<int>& profit) {
        int n = startTime.size();
        std::vector<Job> jobs(n);
        for (int i = 0; i < n; ++i) {
            jobs[i] = {startTime[i], endTime[i], profit[i]};
        }

        std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
            return a.end < b.end;
        });

        std::vector<int> dp(n + 1, 0);

        for (int i = 1; i <= n; ++i) {
            int s = jobs[i - 1].start;
            int p = jobs[i - 1].profit;

            // Binary search for latest job ending <= s
            int low = 0, high = i - 1, prevIdx = 0;
            while (low <= high) {
                int mid = low + (high - low) / 2;
                if (jobs[mid - 1].end <= s) {
                    prevIdx = mid;
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }

            dp[i] = std::max(dp[i - 1], p + dp[prevIdx]);
        }

        return dp[n];
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    static class Job {
        int start, end, profit;
        Job(int s, int e, int p) { start = s; end = e; profit = p; }
    }

    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        int n = startTime.length;
        Job[] jobs = new Job[n];
        for (int i = 0; i < n; i++) {
            jobs[i] = new Job(startTime[i], endTime[i], profit[i]);
        }

        Arrays.sort(jobs, Comparator.comparingInt(a -> a.end));

        int[] dp = new int[n + 1];

        for (int i = 1; i <= n; i++) {
            int s = jobs[i - 1].start;
            int p = jobs[i - 1].profit;

            // Binary search for latest compatible job
            int low = 0, high = i - 1, prevIdx = 0;
            while (low <= high) {
                int mid = low + (high - low) / 2;
                if (jobs[mid - 1].end <= s) {
                    prevIdx = mid;
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }

            dp[i] = Math.max(dp[i - 1], p + dp[prevIdx]);
        }

        return dp[n];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N \log N)$ — Sorting takes $O(N \log N)$ and for each of the $N$ jobs, we perform a binary search costing $O(\log N)$.
- **Space Complexity:** $O(N)$ — For the sorted jobs list and DP array.
