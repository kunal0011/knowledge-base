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
  - sorting
  - amazon
  - google
  - meta
---

# LeetCode 1235: Maximum Profit in Job Scheduling

**Target Companies:** Google (Signature DP Hard), Amazon (Top #1), Meta  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / Binary Search / Sorting  

---

### Problem Statement

We have `n` jobs, where every job is scheduled to be done from `startTime[i]` to `endTime[i]`, obtaining a profit of `profit[i]`.

You're given the `startTime`, `endTime` and `profit` arrays, return the maximum profit you can take such that there are no two jobs in the subset with overlapping time range.

If you choose a job that ends at time `X` you will be able to start another job that starts at time `X`.

---

### Input & Output Formats & Constraints

- **Input:**
  - `startTime: List[int]`
  - `endTime: List[int]`
  - `profit: List[int]`
- **Output:** An integer representing the maximum total profit.
- **Constraints:**
  - $1 \le \text{startTime.length} == \text{endTime.length} == \text{profit.length} \le 5 \times 10^4$
  - $1 \le \text{startTime}[i] < \text{endTime}[i] \le 10^9$
  - $1 \le \text{profit}[i] \le 10^4$

---

### Key Idea & Intuition

#### Weighted Interval Scheduling Formulation
In unweighted interval scheduling (finding the maximum number of non-overlapping intervals, e.g. LC 435), a greedy strategy selecting the earliest end time is optimal. However, when intervals have arbitrary weights/profits, greedy fails because a single high-value long job may yield more profit than multiple short jobs.

#### Dynamic Programming with Binary Search
1. **Sort by End Times:**
   - Group the inputs into triples $(start, end, profit)$ and sort them in ascending order of $end$.
2. **Subproblem Definition:**
   - Let $dp[i]$ be the maximum profit achievable considering a subset of the first $i$ jobs ($1 \le i \le n$).
3. **Transition Decisions for Job $i$:**
   - **Option A (Skip Job $i$):** We do not execute job $i$. The profit is simply $dp[i - 1]$.
   - **Option B (Take Job $i$):** We execute job $i$, earning $profit_i$. Any previously scheduled jobs must have ended $\le start_i$.
   - Using **Binary Search** (`bisect_right` on end times), find the latest job index $j$ ($j < i$) whose end time is $\le start_i$:
     $$dp[i] = \max(dp[i - 1], profit_i + dp[j])$$
4. This yields an overall runtime of $\mathcal{O}(N \log N)$.

---

### Solution Approach (Step-by-Step)

1. **Pack and Sort:**
   - Create list `jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])`.
   - Extract `end_times = [j[1] for j in jobs]`.
2. **Initialize DP Table:**
   - Create array `dp` of size $n + 1$ initialized to 0.
3. **Iterative Evaluation:**
   - For $i$ from 1 to $n$:
     - $s, e, p = jobs[i - 1]$
     - Find the latest non-overlapping job index using binary search:
       `prev_idx = bisect_right(end_times, s, 0, i - 1)`
     - $dp[i] = \max(dp[i - 1], p + dp[prev\_idx])$
4. **Return:**
   - Return $dp[n]$.

---

### Visual Algorithm Walkthrough

#### Trace for `startTime = [1,2,3,3]`, `endTime = [3,4,5,6]`, `profit = [50,10,40,70]`
```
Jobs sorted by endTime:
Job 0: [1, 3], profit = 50
Job 1: [2, 4], profit = 10
Job 2: [3, 5], profit = 40
Job 3: [3, 6], profit = 70

DP Array Evolution:
dp[0] = 0

i = 1 (Job 0: [1, 3], p = 50):
  Latest job ending <= 1: None -> j = 0
  dp[1] = max(dp[0], 50 + dp[0]) = 50

i = 2 (Job 1: [2, 4], p = 10):
  Latest job ending <= 2: None -> j = 0
  dp[2] = max(dp[1], 10 + dp[0]) = max(50, 10) = 50

i = 3 (Job 2: [3, 5], p = 40):
  Latest job ending <= 3: Job 0 ends at 3 <= 3 -> j = 1
  dp[3] = max(dp[2], 40 + dp[1]) = max(50, 40 + 50) = 90

i = 4 (Job 3: [3, 6], p = 70):
  Latest job ending <= 3: Job 0 ends at 3 <= 3 -> j = 1
  dp[4] = max(dp[3], 70 + dp[1]) = max(90, 70 + 50) = 120

Max Profit: 120 (Choosing Job 0 and Job 3).
```

---

### Solved Examples with Multiple Inputs

| `startTime` | `endTime` | `profit` | Compatible Chains Evaluated | Output |
|---|---|---|---|---|
| `[1,2,3,3]` | `[3,4,5,6]` | `[50,10,40,70]` | Jobs 0 & 3: $50 + 70 = 120$ | `120` |
| `[1,2,3,4,6]` | `[3,5,10,6,9]` | `[20,20,100,70,60]` | Job 2: $100$; Jobs 0, 3, 4: $20+70+60 = 150$ | `150` |
| `[1,1,1]` | `[2,3,4]` | `[5,6,4]` | All overlap: pick single max job (Job 1) | `6` |

---

### Multi-Language Implementations

#### Python 3
```python
from bisect import bisect_right

class Solution:
    def jobScheduling(self, startTime: list[int], endTime: list[int], profit: list[int]) -> int:
        jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])
        n = len(jobs)
        end_times = [j[1] for j in jobs]
        
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            s, e, p = jobs[i - 1]
            # Find the latest job that ends before or at current job's start time
            prev_idx = bisect_right(end_times, s, 0, i - 1)
            dp[i] = max(dp[i - 1], p + dp[prev_idx])
            
        return dp[n]
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

struct Job {
    int start;
    int end;
    int profit;
};

class Solution {
public:
    int jobScheduling(std::vector<int>& startTime, std::vector<int>& endTime, std::vector<int>& profit) {
        int n = static_cast<int>(startTime.size());
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

            // Binary search for the latest job ending <= s
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

#### Java 17
```java
import java.util.Arrays;
import java.util.Comparator;

class Solution {
    static class Job {
        int start, end, profit;
        Job(int s, int e, int p) {
            this.start = s;
            this.end = e;
            this.profit = p;
        }
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

            // Binary search for latest compatible job ending <= s
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

- **Time Complexity:** $\mathcal{O}(N \log N)$, where $N$ is the number of jobs. Sorting the $N$ jobs takes $\mathcal{O}(N \log N)$ time. For each of the $N$ jobs, binary searching for the latest compatible job takes $\mathcal{O}(\log N)$ time. Total time is strictly $\mathcal{O}(N \log N)$.
- **Space Complexity:** $\mathcal{O}(N)$ auxiliary space for the sorted jobs array and the 1D DP table.

---

### Takeaway Pattern & Interview Traps

1. **Why Sort by End Time, Not Start Time?** Sorting by end time ensures that when considering job $i$, all potential compatible preceding jobs have already been evaluated and their optimal profits computed, enabling direct binary search over completed prefixes.
2. **Back-to-Back Start/End Compatibility:** The problem explicitly states that if a job ends at time $X$, another job starting at time $X$ is compatible. Thus, we search for `end <= start` (using `bisect_right`), not `end < start`.
3. **1-Indexed DP Array Convenience:** Using an array of size $n + 1$ with $dp[0] = 0$ makes binary search edge cases (where no compatible job exists) seamlessly return index 0 without needing conditional branches.
