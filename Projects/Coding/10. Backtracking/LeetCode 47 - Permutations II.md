---
date: "2025-12-14"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 47: Permutations II"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 47: Permutations II

LeetCode 47 (Permutations II).

The key to this problem is preventing duplicate branches in our backtracking tree. We do this by sorting the input first and using a specific condition to skip duplicates.

Python Implementation

1 from typing import List

2

3 class Solution:

4     def permuteUnique(self, nums: List[int]) -> List[List[int]]:

5         results = []

6         nums.sort()  # Step 1: Sort to handle duplicates easily

7         used = [False] \* len(nums) # Track which indices are currently in the path

8

9         def backtrack(path):

10             # Base Case: If the path is the same length as nums, we found a permutation

11             if len(path) == len(nums):

12                 results.append(path[:]) # Append a copy of the path

13                 return

14

15             for i in range(len(nums)):

16                 # Skip logic:

17                 # 1. If this index is already used in the current branch.

18                 # 2. If this number is the same as the previous number AND

19                 #    the previous number was NOT used. This means we just

20                 #    backtracked from the previous number and are about to

21                 #    start a duplicate branch.

22                 if used[i]:

23                     continue

24                 if i > 0 and nums[i] == nums[i-1] and not used[i-1]:

25                     continue

26

27                 # Action: Choose the number

28                 used[i] = True

29                 path.append(nums[i])

30

31                 # Recurse: Go deeper

32                 backtrack(path)

33

34                 # Backtrack: Undo the choice (pop and mark unused)

35                 path.pop()

36                 used[i] = False

37

38         backtrack([])

39         return results

—

Here is the trace for **Input: `[1, 2, 2]`**.

To make this clear, we will label the duplicate 2s as $2\_A$ (index 1) and $2\_B$ (index 2).

The sorted array is [1, 2\_A, 2\_B].

### The Pruning Logic Recap

Remember the rule: **You cannot pick $2\_B$ if $2\_A$ exists, is currently available (not used), and came before it.** We enforce a strict relative order: $2\_A$ must always be used before $2\_B$ in any specific recursion level.

---

### The Backtracking Tree Diagram

Plaintext

```python
ROOT: []
|
+--- Branch 1: Pick 1 (Index 0)
|    State: [1], used=[T, F, F]
|    |
|    +--- Branch 1.1: Pick 2_A (Index 1)
|    |    State: [1, 2_A], used=[T, T, F]
|    |    |
|    |    +--- Branch 1.1.1: Pick 2_B (Index 2)
|    |         State: [1, 2_A, 2_B] -> ✅ Solution 1 Found
|    |         (Backtrack: unpick 2_B, unpick 2_A)
|    |
|    +--- Branch 1.2: Pick 2_B (Index 2)
|         CHECK: Is 2_B == 2_A? (Yes)
|         CHECK: Is 2_A used? (No)
|         ❌ PRUNE!
|         (Logic: Inside this sub-branch starting with [1], we must pick
|          2_A before 2_B. We cannot skip 2_A.)
|
+--- Branch 2: Pick 2_A (Index 1)
|    State: [2_A], used=[F, T, F]
|    |
|    +--- Branch 2.1: Pick 1 (Index 0)
|    |    State: [2_A, 1], used=[T, T, F]
|    |    |
|    |    +--- Branch 2.1.1: Pick 2_B (Index 2)
|    |         State: [2_A, 1, 2_B] -> ✅ Solution 2 Found
|    |         (Backtrack)
|    |
|    +--- Branch 2.2: Pick 2_B (Index 2)
|         State: [2_A, 2_B], used=[F, T, T]
|         (Note: We CAN pick 2_B here because 2_A is marked True/Used!)
|         |
|         +--- Branch 2.2.1: Pick 1 (Index 0)
|              State: [2_A, 2_B, 1] -> ✅ Solution 3 Found
|              (Backtrack: unpick 1, unpick 2_B, unpick 2_A)
|
+--- Branch 3: Pick 2_B (Index 2)
     CHECK: Is 2_B == 2_A? (Yes)
     CHECK: Is 2_A used? (No - we just backtracked from Branch 2)
     ❌ PRUNE!
     (Logic: At the root level, we already created all permutations starting
      with 2 using 2_A. We don't need to do it again with 2_B.)
```

### Detailed Navigation Walkthrough

**1. The Root Level Loop ($i=0, 1, 2$)**

* **$i=0$ (Value 1):** We pick `1`. We dive down.

  * Inside this branch, we are left with `[2_A, 2_B]`.
  * We pick `2_A` first. Then `2_B`. Result: `[1, 2, 2]`.
  * We backtrack. Now we try to pick `2_B`.
  * **The Check:** We see `2_B` is same as `2_A`. `2_A` is NOT used right now. **STOP.**
  * *Why?* If we picked `2_B`, we would form `[1, 2_B...]`. Since `2_A` and `2_B` are identical values, `[1, 2_B...]` would look exactly the same as `[1, 2_A...]` which we just finished.
* **$i=1$ (Value 2\_A):** We pick `2_A`.

  * Now `used = [F, T, F]`.
  * We recurse. We can pick `1` -> `[2, 1, 2]`.
  * We recurse. We can pick `2_B`.
  * **Wait, why is picking `2_B` allowed here?**

    * Because the condition is `if nums[i] == nums[i-1] and not used[i-1]`.
    * Here, `nums[i-1]` is `2_A`. Is `2_A` used? **YES** (from the parent call).
    * So the `not used` condition fails. We are allowed to proceed.
    * Result: `[2, 2, 1]`.
* **$i=2$ (Value 2\_B):** We are back at Root.

  * We try to pick `2_B` as the *start* of the permutation.
  * Check: `nums[2] == nums[1]` (`2_B == 2_A`).
  * Check: Is `2_A` used? **NO**. (It is `False` at the root level).
  * **PRUNE.** We skip this entire branch.

### Final Results

The code successfully generates:

1. `[1, 2, 2]`
2. `[2, 1, 2]`
3. `[2, 2, 1]`

And it successfully skipped the duplicate calculations that would have occurred if we treated `2_A` and `2_B` as different numbers.

The Tree Diagram & Navigation Explained

Let's trace the execution with Input: `[1, 1, 2]`.

We will distinguish the two 1s by their index: 1a (index 0) and 1b (index 1).

Input: [1a, 1b, 2]

Legend:

\* [ ... ] : Current Path

\* ❌ : Pruned (skipped to avoid duplicate)

\* ✅ : Valid Permutation found

1 ROOT: []

2 |

3 +--- Branch 1: Pick 1a (index 0)

4 |    Path: [1a]

5 |    Used: [T, F, F]

6 |    |

7 |    +--- Pick 1b (index 1)

8 |    |    Path: [1a, 1b]

9 |    |    Used: [T, T, F]

10 |    |    |

11 |    |    +--- Pick 2 (index 2)

12 |    |         Path: [1a, 1b, 2] -> ✅ FOUND RESULT #1

13 |    |         (Backtrack: pop 2, used[2]=False)

14 |    |

15 |    +--- (Backtrack: pop 1b, used[1]=False)

16 |    |

17 |    +--- Pick 2 (index 2)

18 |         Path: [1a, 2]

19 |         Used: [T, F, T]

20 |         |

21 |         +--- Pick 1b (index 1)

22 |              Path: [1a, 2, 1b] -> ✅ FOUND RESULT #2

23 |              (Backtrack...)

24 |

25 +--- (Backtrack: pop 1a, used[0]=False)

26 |

27 +--- Branch 2: Pick 1b (index 1)

28 |    Attempting to Pick 1b...

29 |    CHECK: nums[1] == nums[0]? YES.

30 |    CHECK: used[0] == False? YES (We just backtracked from 1a).

31 |    ❌ PRUNED.

32 |    Explanation: We just finished the tree rooted at 1a.

33 |    Since 1b is the same value, a tree rooted at 1b would be identical.

34 |

35 +--- Branch 3: Pick 2 (index 2)

36      Path: [2]

37      Used: [F, F, T]

38      |

39      +--- Pick 1a (index 0)

40      |    Path: [2, 1a]

41      |    Used: [T, F, T]

42      |    |

43      |    +--- Pick 1b (index 1)

44      |         Path: [2, 1a, 1b] -> ✅ FOUND RESULT #3

45      |         (Backtrack...)

46      |

47      +--- (Backtrack from 1a...)

48      |

49      +--- Pick 1b (index 1)

50           Attempting to Pick 1b...

51           CHECK: nums[1] == nums[0]? YES.

52           CHECK: used[0] == False? YES.

53           ❌ PRUNED.

54           Explanation: Even deep in the tree, if we have options {1a, 1b}

55           and we skip 1a to pick 1b, we create a duplicate.

Visual Summary of the "Pruning" Logic

The critical line is if i > 0 and nums[i] == nums[i-1] and not used[i-1]: continue.

Think of it this way: We enforce that duplicates must be used in their original order.

If you have three 5s: 5a, 5b, 5c.

\* You are allowed to generate [5a, 5b, 5c].

\* You are NOT allowed to generate [5b, 5a, 5c].

When the loop reaches 5b (index i), it checks 5a (index i-1).

If 5a is not used, it means we intentionally skipped 5a to get to 5b. We forbid this. We force the algorithm to use 5a before it is allowed to use 5b.