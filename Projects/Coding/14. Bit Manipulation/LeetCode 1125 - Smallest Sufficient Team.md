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
  - dynamic-programming
  - amazon
  - google
---

# LeetCode 1125: Smallest Sufficient Team

**Target Companies:** Google, Amazon, Meta, Uber  
**Difficulty:** Hard  
**Topic:** Bit Manipulation / Dynamic Programming (Exact Cover via Bitmask DP)

---

### Problem Statement

In a project, you have a list of required skills `req_skills`, and a list of people. The $i$-th person `people[i]` contains a list of skills that the person has.

Consider a sufficient team: a set of people such that for every required skill in `req_skills`, there is at least one person in the team who has that skill. We can represent these teams by the index of each person.

* For example, `team = [0, 1, 3]` represents the people with indices $0$, $1$, and $3$.

Return *any sufficient team of the smallest possible size, represented by the index of each person*. You may return the answer in **any order**.

It is guaranteed an answer exists.

---

### Input & Output Formats & Constraints

- **Input:**
  - `req_skills`: `List[str]`, where $1 \le \text{req\_skills.length} \le 16$. Each string consists of lowercase English letters and $1 \le \text{req\_skills}[i]\text{.length} \le 16$. All strings in `req_skills` are pairwise unique.
  - `people`: `List[List[str]]`, where $1 \le \text{people.length} \le 60$. $0 \le \text{people}[i]\text{.length} \le 16$. Every skill possessed by `people[i]` is in `req_skills`, and within `people[i]`, skills are unique.
- **Output:**
  - `List[int]`: A list of 0-based indices representing the smallest subset of people whose union of skills contains all skills in `req_skills`.
- **Constraints:**
  - The number of required skills $n \le 16$.
  - The number of candidates $m \le 60$.
  - An answer is guaranteed to exist.

---

### Key Idea & Intuition

The problem asks for the minimum size subset of people whose skill sets union up to the full universe of skills `req_skills`. In computational complexity theory, this is the NP-hard **Set Cover Problem**.

However, notice the crucial constraint:
$$\text{Number of required skills } n \le 16$$

When the universe size $n \le 16$, the total number of distinct skill subsets is $2^{16} = 65,536$. This is small enough to permit **Dynamic Programming over Bitmasks**:
1. Assign each skill $s_j \in \text{req\_skills}$ an index $j \in \{0, 1, \dots, n - 1\}$.
2. Represent any set of skills as an integer mask in $[0, 2^n - 1]$, where the $j$-th bit is $1$ if and only if skill $s_j$ is possessed.
3. For each candidate person $i \in [0, m - 1]$, compute their skill bitmask $p\_mask_i = \sum_{s \in \text{people}[i]} 2^{\text{skill\_index}[s]}$.
4. Let $dp[mask]$ store the minimal team (or minimal team bitmask / size with parent pointers) that covers the skill subset represented by $mask$.
5. When considering candidate $i$ with skill set $p\_mask_i$:
   $$\forall \text{ reachable } curr\_mask: \quad new\_mask = curr\_mask \mid p\_mask_i$$
   If $|team(curr\_mask)| + 1 < |team(new\_mask)|$, we relax:
   $$team(new\_mask) = team(curr\_mask) \cup \{i\}$$
6. Since $m \le 60$, a 64-bit integer `long long` / `long` can hold a bitmask of selected people (since $60 < 64$). Thus, `team[mask]` can be stored directly as a single 64-bit integer, where bit $k$ is 1 if person $k$ is recruited!

---

### Solution Approach (Step-by-Step)

1. **Mapping Skills to Bit Positions:**
   - Construct a hash map `skill_to_id` mapping each skill string to an index $0, 1, \dots, n-1$.
   - Target mask $target = (1 \ll n) - 1$.

2. **Encode Candidate Skills:**
   - For each person $i \in [0, m-1]$, compute `p_mask = sum of (1 << skill_to_id[skill])`.
   - If `p_mask == 0`, the candidate provides no required skills and can be safely ignored.

3. **Initialize DP Table:**
   - Let `team[mask]` be the 64-bit mask of people used to cover skill set `mask`.
   - Initialize `team[0] = 0` (0 people cover empty skill set). All other masks $1 \dots 2^n - 1$ are unreached (set to a sentinel value or tracked with a size array initialized to $\infty$).

4. **Process Candidates Sequentially:**
   - For each person $i$:
     - For each currently reachable skill mask $curr\_mask$:
       - Let $new\_mask = curr\_mask \mid p\_mask$.
       - The candidate team would be `team[curr_mask] | (1ULL << i)`.
       - If $new\_mask$ is unvisited OR $\text{popcount}(\text{team}[curr\_mask]) + 1 < \text{popcount}(\text{team}[new\_mask])$:
         - Update $\text{team}[new\_mask] = \text{team}[curr\_mask] \mid (1\text{ULL} \ll i)$.

5. **Reconstruct Final Team:**
   - For target mask $target = (1 \ll n) - 1$, inspect the bits of `team[target]`.
   - For each bit $i \in [0, m-1]$ that is set to $1$, append index $i$ to the result array.

---

### Visual Algorithm Walkthrough

Suppose `req_skills = ["java", "nodejs", "reactjs"]` ($n = 3$, target mask $= 111_2 = 7$).
- `"java"` $\to$ bit 0 ($1$), `"nodejs"` $\to$ bit 1 ($2$), `"reactjs"` $\to$ bit 2 ($4$).

Candidates:
- Person 0: `["java"]` $\implies p\_mask = 001_2$
- Person 1: `["nodejs"]` $\implies p\_mask = 010_2$
- Person 2: `["nodejs", "reactjs"]` $\implies p\_mask = 110_2$

```
Target Mask: 111 (all 3 skills)

Initial DP State:
dp[000] = {} (size 0)

---------------------------------------------------------------------------------
Person 0 (mask = 001):
  Combine with dp[000]:
    new_mask = 000 | 001 = 001
    dp[001] = {Person 0} (size 1)

---------------------------------------------------------------------------------
Person 1 (mask = 010):
  Combine with dp[000]:
    new_mask = 000 | 010 = 010
    dp[010] = {Person 1} (size 1)
  Combine with dp[001]:
    new_mask = 001 | 010 = 011
    dp[011] = {Person 0, Person 1} (size 2)

---------------------------------------------------------------------------------
Person 2 (mask = 110):
  Combine with dp[000]:
    new_mask = 000 | 110 = 110
    dp[110] = {Person 2} (size 1)
  Combine with dp[001]:
    new_mask = 001 | 110 = 111  <-- TARGET REACHED!
    dp[111] = {Person 0, Person 2} (size 2)
  Combine with dp[010]:
    new_mask = 010 | 110 = 110 (size 2 > existing size 1, ignore)
  Combine with dp[011]:
    new_mask = 011 | 110 = 111 (size 3 > existing size 2, ignore)

---------------------------------------------------------------------------------
Final Result:
  dp[111] = {Person 0, Person 2}
  Return [0, 2]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Case (Overlapping Skill Sets)

- **Input:**
  - `req_skills = ["java", "nodejs", "reactjs"]`
  - `people = [["java"], ["nodejs"], ["nodejs", "reactjs"]]`
- **Tracing Transitions:**

| Iteration | Person | $p\_mask$ | New Mask Reached | Updated Team | Team Size |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0 | `001` | `001` | `[0]` | 1 |
| 1 | 1 | `010` | `010`<br>`011` | `[1]`<br>`[0, 1]` | 1<br>2 |
| 2 | 2 | `110` | `110`<br>`111` | `[2]`<br>`[0, 2]` | 1<br>2 |

- **Output:** `[0, 2]`

#### Example 2: Competing Candidate Teams (Tie-Breaking by Minimum Size)

- **Input:**
  - `req_skills = ["algorithms", "math", "java", "reactjs", "csharp", "aws"]` ($n=6$)
  - `people = [["algorithms","math","java"], ["algorithms","reactjs","csharp"], ["reactjs","csharp","aws"], ["math","aws"]]`
- **Evaluation:**
  - Team $[0, 2]$ has skills:
    - Person 0: `algorithms`, `math`, `java`
    - Person 2: `reactjs`, `csharp`, `aws`
    - Union: all 6 required skills!
    - Size: 2 people.
- **Output:** `[0, 2]`

---

### Multi-Language Implementations

#### Python 3

```python
from typing import List

class Solution:
    def smallestSufficientTeam(self, req_skills: List[str], people: List[List[str]]) -> List[int]:
        n = len(req_skills)
        skill_map = {skill: i for i, skill in enumerate(req_skills)}
        target_mask = (1 << n) - 1
        
        # dp[mask] = list of candidate indices that cover 'mask'
        dp = {0: []}
        
        for p_idx, p_skills in enumerate(people):
            # Compute current candidate's skill bitmask
            p_mask = 0
            for skill in p_skills:
                if skill in skill_map:
                    p_mask |= (1 << skill_map[skill])
                    
            if p_mask == 0:
                continue
                
            # Iterate through snapshot of existing achieved skill combinations
            for curr_mask, team in list(dp.items()):
                new_mask = curr_mask | p_mask
                if new_mask not in dp or len(team) + 1 < len(dp[new_mask]):
                    dp[new_mask] = team + [p_idx]
                    
        return dp[target_mask]
```

#### C++17

```cpp
#include <vector>
#include <string>
#include <unordered_map>
#include <climits>

class Solution {
public:
    std::vector<int> smallestSufficientTeam(const std::vector<std::string>& req_skills,
                                           const std::vector<std::vector<std::string>>& people) {
        int n = static_cast<int>(req_skills.size());
        int m = static_cast<int>(people.size());
        
        std::unordered_map<std::string, int> skill_map;
        for (int i = 0; i < n; ++i) {
            skill_map[req_skills[i]] = i;
        }
        
        int total_states = 1 << n;
        // team_mask[mask] stores a 64-bit integer representing the chosen people
        // Since m <= 60, all person indices 0..59 fit inside unsigned long long.
        std::vector<unsigned long long> team_mask(total_states, 0);
        std::vector<int> dp_size(total_states, 1e9);
        
        dp_size[0] = 0;
        
        for (int i = 0; i < m; ++i) {
            int p_mask = 0;
            for (const std::string& skill : people[i]) {
                auto it = skill_map.find(skill);
                if (it != skill_map.end()) {
                    p_mask |= (1 << it->second);
                }
            }
            if (p_mask == 0) continue;
            
            for (int curr = 0; curr < total_states; ++curr) {
                if (dp_size[curr] == 1e9) continue;
                
                int next_mask = curr | p_mask;
                if (dp_size[curr] + 1 < dp_size[next_mask]) {
                    dp_size[next_mask] = dp_size[curr] + 1;
                    team_mask[next_mask] = team_mask[curr] | (1ULL << i);
                }
            }
        }
        
        // Reconstruct the team from 64-bit mask
        unsigned long long best_team = team_mask[total_states - 1];
        std::vector<int> result;
        for (int i = 0; i < m; ++i) {
            if ((best_team >> i) & 1ULL) {
                result.push_back(i);
            }
        }
        return result;
    }
};
```

#### Java

```java
import java.util.*;

public class Solution {
    public int[] smallestSufficientTeam(String[] reqSkills, List<List<String>> people) {
        int n = reqSkills.length;
        int m = people.size();
        
        Map<String, Integer> skillMap = new HashMap<>();
        for (int i = 0; i < n; i++) {
            skillMap.put(reqSkills[i], i);
        }
        
        int totalStates = 1 << n;
        // teamMask[mask] stores a 64-bit long representing chosen candidates
        long[] teamMask = new long[totalStates];
        int[] dpSize = new int[totalStates];
        Arrays.fill(dpSize, 1000); // sentinel infinity
        dpSize[0] = 0;
        
        for (int i = 0; i < m; i++) {
            int pMask = 0;
            for (String skill : people.get(i)) {
                Integer idx = skillMap.get(skill);
                if (idx != null) {
                    pMask |= (1 << idx);
                }
            }
            if (pMask == 0) continue;
            
            for (int curr = 0; curr < totalStates; curr++) {
                if (dpSize[curr] == 1000) continue;
                
                int nextMask = curr | pMask;
                if (dpSize[curr] + 1 < dpSize[nextMask]) {
                    dpSize[nextMask] = dpSize[curr] + 1;
                    teamMask[nextMask] = teamMask[curr] | (1L << i);
                }
            }
        }
        
        long bestTeam = teamMask[totalStates - 1];
        int teamCount = Long.bitCount(bestTeam);
        int[] result = new int[teamCount];
        int ptr = 0;
        for (int i = 0; i < m; i++) {
            if (((bestTeam >> i) & 1L) == 1L) {
                result[ptr++] = i;
            }
        }
        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(m \cdot 2^n)$
  - Mapping skills takes $\mathcal{O}(\sum \text{len}(req\_skills))$.
  - Encoding candidate skills takes $\mathcal{O}(\sum \text{len}(people[i]))$.
  - The DP transitions examine at most $2^n$ states per candidate. With $n \le 16$ and $m \le 60$, $m \cdot 2^n \le 60 \times 65,536 \approx 3.9 \times 10^6$ operations, which executes in $< 35\text{ ms}$.
- **Space Complexity:** $\mathcal{O}(2^n)$
  - The DP array stores $2^n$ states. With $2^{16} = 65,536$ integers (or 64-bit integers), this consumes $\approx 512\text{ KB}$, which easily fits into L2/L3 cache.

---

### Takeaway Pattern & Interview Traps

1. **NP-Hard Subset Cover with Small Universe ($N \le 16$):**
   - Whenever an interview problem models Exact Cover or Set Cover where the universe cardinality $N \le 16$ or $N \le 20$, immediately consider **Bitmask DP**.
2. **64-bit Bitmask for Set of Indices:**
   - Because the number of candidates $M \le 60 \le 64$, we can represent candidate subsets as a single 64-bit unsigned integer (`long long` in C++, `long` in Java). This eliminates the need for expensive list allocations and pointer chasing during DP relaxation!
3. **Loop Direction / Snapshotting:**
   - In Python, iterating over `list(dp.items())` snapshots the reachable masks before appending new ones.
   - In C++ / Java, because a candidate cannot be picked twice for the same team update, transitioning `curr | p_mask` into a separate state or checking DP sizes ensures no self-duplication occurs within the same candidate step.