---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 316: Remove Duplicate Letters"
tags:
  - leetcode
  - coding
  - stack
---

# LeetCode 316: Remove Duplicate Letters

Below is a structured, interview-grade explanation of **LeetCode 316 – Remove Duplicate Letters**, aligned with how this problem is typically expected to be reasoned and implemented.

---

## 1. Problem Statement

**LeetCode 316 – Remove Duplicate Letters**

Given a string `s`, remove duplicate letters so that **every letter appears exactly once** and the resulting string is the **smallest in lexicographical order** among all possible results.

**Constraints**

* `1 ≤ len(s) ≤ 10^4`
* `s` consists of lowercase English letters.

**Example**

```text
Input:  s = "cbacdcbc"
Output: "acdb"
```

---

## 2. Key Observation

1. **Uniqueness constraint**  
   Each character must appear **exactly once** in the result.
2. **Lexicographical minimality**  
   Among all subsequences containing all unique characters, we must pick the **lexicographically smallest** one.
3. **Subsequence, not permutation**  
   The relative order of characters must follow the original string.
4. **Greedy choice with reversibility**  
   If a character currently in the result can be replaced by a smaller character **and** it appears again later, we should remove it now and reinsert it later.

This combination of **greedy decision + ability to undo earlier choices** strongly suggests a **monotonic stack**.

---

## 3. Stack Key Insight (Core Greedy Logic)

We maintain:

* A **stack** that stores the current result characters
* A **frequency map** (`count`) to know if a character appears again later
* A **visited set** to ensure uniqueness

### When processing a character `c`:

1. If `c` is already in the stack → **skip it**
2. Otherwise:

   * While all three conditions hold:

     * Stack is not empty
     * Top of stack `> c` (lexicographically larger)
     * Top of stack appears again later (`count[top] > 0`)
   * Pop the stack (we can safely reinsert it later)
3. Push `c` onto the stack and mark it visited

This guarantees:

* Lexicographically smallest order
* Each character appears once
* Valid subsequence

---

## 4. Python 3 Solution (with typing)

```python
from typing import Dict, List
from collections import Counter

class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        count: Dict[str, int] = Counter(s)
        visited = set()
        stack: List[str] = []

        for ch in s:
            # Decrease remaining count
            count[ch] -= 1

            # If already in stack, skip
            if ch in visited:
                continue

            # Maintain lexicographical order
            while stack and stack[-1] > ch and count[stack[-1]] > 0:
                removed = stack.pop()
                visited.remove(removed)

            stack.append(ch)
            visited.add(ch)

        return "".join(stack)
```

**Time Complexity**: `O(n)`  
Each character is pushed and popped at most once.

**Space Complexity**: `O(1)`  
At most 26 lowercase letters in stack and sets.

---

## 5. Worked-Out Example (Step-by-Step)

### Input

```
s = "cbacdcbc"
```

### Initial State

```
count = {c:4, b:2, a:1, d:1}
stack = []
visited = {}
```

---

### Step-by-Step Processing

| Char | Action | Stack | Reason |
| --- | --- | --- | --- |
| c | push | [c] | empty stack |
| b | pop c, push b | [b] | b < c and c appears later |
| a | pop b, push a | [a] | a < b and b appears later |
| c | push | [a, c] | c > a |
| d | push | [a, c, d] | d > c |
| c | skip | [a, c, d] | already visited |
| b | cannot pop d | [a, c, d, b] | d does NOT appear again |
| c | skip | [a, c, d, b] | already visited |

---

### Final Stack

```
[a, c, d, b]
```

### Output

```
"acdb"
```

---

## 6. Why This Greedy Strategy Works

* We only remove a character if:

  * It improves lexicographical order
  * We are guaranteed to see it again later
* Once a character is popped, it can be reinserted later at a better position
* The stack remains **monotonic increasing where possible**

This is a **canonical monotonic stack + greedy** problem.

---

## 7. Related Pattern

* **Monotonic Stack (Greedy)**
* Similar to:

  * LeetCode 1081 (same problem statement)
  * LeetCode 402 (Remove K Digits)
  * LeetCode 321 (Create Maximum Number)

If you want, I can also:

* Provide a **formal greedy proof**
* Compare with **LeetCode 1081**
* Draw a **stack evolution diagram** for intuition