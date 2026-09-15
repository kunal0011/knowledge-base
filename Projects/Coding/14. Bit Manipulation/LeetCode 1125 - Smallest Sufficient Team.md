---
date: "2026-08-29"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 1125: Smallest Sufficient Team"
tags:
  - leetcode
  - coding
  - bit-manipulation
---

# LeetCode 1125: Smallest Sufficient Team

**Target Companies:** Google (Hard), Amazon

---

### Problem Statement

Given `req_skills` and a list of `people` with skills, find the smallest team that collectively possesses all required skills.

---

### Key Observation

* Since `req_skills <= 16`, represent skill sets as integer Bitmasks.
* Target mask: `(1 << len(req_skills)) - 1`.
* DP state: `dp[skill_mask] = smallest_list_of_person_indices` that achieve this skill combination.
* For each person with skill mask `p_mask`, update: `dp[curr_mask | p_mask] = min_size(dp[curr_mask | p_mask], dp[curr_mask] + [p_idx])`.

---

### Core Technique: Bitmask Dynamic Programming (Subset Cover)

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def smallestSufficientTeam(self, req_skills: List[str], people: List[List[str]]) -> List[int]:
        n = len(req_skills)
        skill_map = {skill: i for i, skill in enumerate(req_skills)}
        target_mask = (1 << n) - 1
        
        # dp[mask] = list of person indices
        dp = {0: []}
        
        for p_idx, p_skills in enumerate(people):
            # Calculate this person's skill bitmask
            p_mask = 0
            for skill in p_skills:
                if skill in skill_map:
                    p_mask |= (1 << skill_map[skill])
                    
            if p_mask == 0:
                continue
                
            # Try combining with all previously reached skill sets
            for curr_mask, team in list(dp.items()):
                new_mask = curr_mask | p_mask
                if new_mask not in dp or len(team) + 1 < len(dp[new_mask]):
                    dp[new_mask] = team + [p_idx]
                    
        return dp[target_mask]
```

---

### Worked-Out Example

```python
req_skills = ["java","nodejs","reactjs"] (3 skills, target = 7 (111_2))
Person 0: java (001) -> dp[1] = [0]
Person 1: nodejs (010) -> dp[2] = [1], dp[3] = [0, 1]
Person 2: nodejs, reactjs (110) -> dp[6] = [2], dp[7] = [0, 2]
Result team = [0, 2] (length 2)
```

---

### Complexity Analysis

* **Time Complexity:** `O(num_people * 2^num_skills)`
* **Space Complexity:** `O(2^num_skills)`

---

### Takeaway Pattern

Represent requirement sets as bitmasks to transform NP-Hard set cover into optimal Bitmask DP.