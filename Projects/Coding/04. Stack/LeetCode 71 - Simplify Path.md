---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 71: Simplify Path"
tags:
  - leetcode
  - coding
  - stack
---

# LeetCode 71: Simplify Path

Below is a structured, interview-ready explanation of **LeetCode 71 – Simplify Path**, aligned with how stack problems are typically reasoned about.

---

## LeetCode 71 — Simplify Path

### Problem Statement

You are given an **absolute Unix-style file path** as a string `path`.  
Your task is to **simplify** it and return the **canonical path**.

A canonical path must follow these rules:

1. It starts with a single `/`.
2. Directories are separated by exactly one `/`.
3. `.` refers to the current directory → ignore it.
4. `..` refers to the parent directory → remove the last valid directory if possible.
5. Multiple consecutive slashes `/` are treated as a single slash.
6. The path must **not end with a slash**, unless it is the root `/`.

---

### Key Observations

1. The path is **absolute**, so we always start from root `/`.
2. Only **three special tokens** need handling:

   * `"."` → no effect
   * `".."` → go one directory up
   * `""` (empty, due to `//`) → ignore
3. Every other token represents a **valid directory name**.
4. The final path is determined solely by the **order of valid directories that remain**.

---

### Why a Stack Works Here (Key Insight)

This is a classic **path navigation** problem.

* When we enter a directory → **push** it.
* When we see `..` → **pop** the last directory (if any).
* When we see `.` or empty → **do nothing**.

This exactly matches **stack behavior (LIFO)**:

* Last directory entered is the first one exited.
* Ensures correct parent-child traversal.

Hence, **stack = current path state**.

---

### Algorithm (High-Level)

1. Split the path using `/`.
2. Initialize an empty stack.
3. Iterate through each part:

   * Ignore `""` and `"."`
   * If `".."` → pop from stack (if not empty)
   * Else → push directory name
4. Join stack contents using `/` and prefix with `/`.

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def simplifyPath(self, path: str) -> str:
        stack: List[str] = []
        
        for part in path.split("/"):
            if part == "" or part == ".":
                continue
            elif part == "..":
                if stack:
                    stack.pop()
            else:
                stack.append(part)
        
        return "/" + "/".join(stack)
```

---

### Worked Example (Step-by-Step)

#### Input

```
path = "/a/./b/../../c/"
```

#### Step 1: Split

```
["", "a", ".", "b", "..", "..", "c", ""]
```

#### Step 2: Process with Stack

| Token | Action | Stack State |
| --- | --- | --- |
| "" | Ignore | [] |
| "a" | Push | ["a"] |
| "." | Ignore | ["a"] |
| "b" | Push | ["a", "b"] |
| ".." | Pop | ["a"] |
| ".." | Pop | [] |
| "c" | Push | ["c"] |
| "" | Ignore | ["c"] |

#### Step 3: Build Result

```
"/" + "c" = "/c"
```

#### Output

```
"/c"
```

---

### Time & Space Complexity

* **Time Complexity:** `O(n)` where `n` is the length of the path string.
* **Space Complexity:** `O(n)` for the stack in the worst case.

---

### Interview Takeaway

* This problem tests **string parsing + stack fundamentals**.
* The key insight is recognizing `".."` as a **backtracking operation**, which maps naturally to stack pop.
* Clean handling of edge cases (`//`, `.`, root traversal) is critical.

If you want, I can also explain:

* Why recursion is unnecessary here
* Common mistakes candidates make
* How this relates to filesystem navigation in real OS kernels