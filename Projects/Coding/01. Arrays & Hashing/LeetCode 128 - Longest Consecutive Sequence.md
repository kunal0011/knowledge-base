---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 128: Longest Consecutive Sequence"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - amazon
  - google
---

# LeetCode 128: Longest Consecutive Sequence

**Target Companies:** Amazon, Google (Top Classic), Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Hash Set / Sequence Start Detection

---

### Problem Statement

Given an unsorted array of integers `nums`, return the length of the **longest consecutive elements sequence**.

You must write an algorithm that runs in **$O(n)$ time**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `int` (maximum sequence length)
- **Constraints:**
  - $0 \le \text{nums.length} \le 10^5$
  - $-10^9 \le \text{nums}[i] \le 10^9$

---

### Key Idea & Intuition

- **Why Not Sort?**
  - Sorting takes $O(N \log N)$, which violates the $O(N)$ strict constraint.
- **Sequence Root Identification:**
  - Insert all elements into a Hash Set in $O(N)$.
  - A number `x` is the **start of a consecutive sequence** if and only if `x - 1` is NOT in the set!
  - If `x - 1` is in the set, skip `x` (it will be counted as part of the sequence starting earlier).
  - If `x - 1` is NOT in the set:
    - Count upwards `x + 1, x + 2, ...` as long as they exist in the set.
  - Because each number is visited at most twice (once in outer loop, once during sequence traversal), total runtime is strictly $O(N)$!

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        max_streak = 0
        
        for num in num_set:
            # Only start counting if num is the beginning of a streak
            if num - 1 not in num_set:
                curr_num = num
                curr_streak = 1
                
                while curr_num + 1 in num_set:
                    curr_num += 1
                    curr_streak += 1
                    
                max_streak = max(max_streak, curr_streak)
                
        return max_streak
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_set>
#include <algorithm>

class Solution {
public:
    int longestConsecutive(std::vector<int>& nums) {
        std::unordered_set<int> numSet(nums.begin(), nums.end());
        int maxStreak = 0;

        for (int num : numSet) {
            if (!numSet.count(num - 1)) {
                int currNum = num;
                int currStreak = 1;

                while (numSet.count(currNum + 1)) {
                    currNum++;
                    currStreak++;
                }

                maxStreak = std::max(maxStreak, currStreak);
            }
        }
        return maxStreak;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.HashSet;
import java.util.Set;

class Solution {
    public int longestConsecutive(int[] nums) {
        Set<Integer> numSet = new HashSet<>();
        for (int num : nums) numSet.add(num);

        int maxStreak = 0;

        for (int num : numSet) {
            if (!numSet.contains(num - 1)) {
                int currNum = num;
                int currStreak = 1;

                while (numSet.contains(currNum + 1)) {
                    currNum++;
                    currStreak++;
                }

                maxStreak = Math.max(maxStreak, currStreak);
            }
        }
        return maxStreak;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Creating set is $O(N)$. Each consecutive sequence is traversed in a forward linear chain; no element is traversed more than once inside the while loop.
- **Space Complexity:** $O(N)$ to store numbers in the hash set.
