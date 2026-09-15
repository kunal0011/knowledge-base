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
  - string
  - amazon
  - google
---

# LeetCode 71: Simplify Path

**Target Companies:** Meta, Google, Amazon, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Stack / String Tokenization / Unix File Path Resolution

---

### Problem Statement

Given an absolute path for a Unix-style file system, which begins with a slash `'/'`, transform this path into its **simplified canonical path**.

In Unix-style file system rules:
- A period `'.'` refers to the current directory.
- A double period `'..'` refers to the directory up a level (the parent directory).
- Multiple consecutive slashes such as `'//'` and `'///'` are treated as a single slash `'/'`.
- Any sequence of periods that does not match `'.'` or `'..'` (e.g. `'...'` or `'....'`) should be treated as valid file/directory names.

The canonical path should have the following format:
1. The path must start with a single slash `'/'`.
2. Directories within the path must be separated by exactly one slash `'/'`.
3. The path must not end with a slash `'/'`, unless it is the root directory.
4. The path must not contain any single or double periods (`'.'` and `'..'`) used to denote current or parent directories.

Return *the simplified canonical path*.

---

### Input & Output Formats & Constraints

- **Input:**
  - `path`: `str`, an absolute Unix-style file path starting with `'/'`.
- **Output:**
  - `str`: The simplified canonical path.
- **Constraints:**
  - $1 \le \text{path.length} \le 3000$.
  - `path` consists of English letters, digits, period `'.'`, slash `'/'` or `'_'`.
  - `path` is a valid absolute Unix path.

---

### Key Idea & Intuition

When navigating directory hierarchies:
- Moving into a sub-directory represents entering a deeper scope.
- `".."` navigates back up one level to the immediate parent directory.
- `"."` stays within the current directory (no-op).
- Empty components arising from consecutive slashes (e.g. `"/a//b"`) are redundant (no-op).

Because `".."` always removes the **most recently entered directory**, this behavior maps directly to a **LIFO Stack**:
1. Split the string by the delimiter `'/'` into directory tokens.
2. For each token:
   - If token is empty `""` or single dot `"."`: ignore it.
   - If token is double dot `".."`: if the stack is non-empty, pop the top directory. If the stack is already empty, we are at root, so do nothing.
   - Otherwise: the token is a legitimate directory name (such as `"home"`, `"..."`, or `"a"`). Push it onto the stack.
3. Finally, join all tokens in the stack with `'/'` and prefix with a leading `'/'`.

---

### Solution Approach (Step-by-Step)

1. Split `path` using `'/'` as the delimiter.
2. Initialize `stack = []`.
3. For each token in the split components:
   - If `token in ("", ".")`: continue.
   - Else if `token == ".."`:
     - If `stack`: `stack.pop()`
   - Else:
     - `stack.append(token)`
4. Return `'/' + '/'.join(stack)`.

---

### Visual Algorithm Walkthrough

Trace `path = "/a/./b/../../c/"`:

```
Split tokens: ["", "a", ".", "b", "..", "..", "c", ""]

1. Token: ""    -> Redundant slash -> Ignore
2. Token: "a"   -> Valid directory -> Push "a"     Stack: ["a"]
3. Token: "."   -> Current directory -> Ignore     Stack: ["a"]
4. Token: "b"   -> Valid directory -> Push "b"     Stack: ["a", "b"]
5. Token: ".."  -> Parent dir -> Pop "b"           Stack: ["a"]
6. Token: ".."  -> Parent dir -> Pop "a"           Stack: []
7. Token: "c"   -> Valid directory -> Push "c"     Stack: ["c"]
8. Token: ""    -> Redundant trailing -> Ignore   Stack: ["c"]

Reassemble canonical path:
  "/" + "/".join(["c"]) = "/c"
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Redundant Traversal

- **Input:** `path = "/home//foo/"`
- **Tokens:** `["", "home", "", "foo", ""]`
- **Output:** `"/home/foo"`

#### Example 2: Going Above Root Directory

- **Input:** `path = "/../"`
- **Tokens:** `["", "..", ""]`
- **Stack:** Popping on empty stack is a no-op; stays at root.
- **Output:** `"/"`

#### Example 3: Valid Triple Periods / Filenames

- **Input:** `path = "/.../a/../b/c/../d/./"`
- **Tokens:** `["", "...", "a", "..", "b", "c", "..", "d", "."]`
- **Step Tracing:**
  - Push `"..."` $\to$ Push `"a"` $\to$ Pop `"a"` $\to$ Push `"b"` $\to$ Push `"c"` $\to$ Pop `"c"` $\to$ Push `"d"`.
  - Stack contains: `["...", "b", "d"]`.
- **Output:** `"/.../b/d"`

---

### Multi-Language Implementations

#### Python 3

```python
from typing import List

class Solution:
    def simplifyPath(self, path: str) -> str:
        stack: List[str] = []

        for token in path.split("/"):
            if token == "" or token == ".":
                continue
            elif token == "..":
                if stack:
                    stack.pop()
            else:
                stack.append(token)

        return "/" + "/".join(stack)
```

#### C++17

```cpp
#include <string>
#include <vector>
#include <sstream>

class Solution {
public:
    std::string simplifyPath(const std::string& path) {
        std::vector<std::string> stack;
        std::stringstream ss(path);
        std::string token;

        while (std::getline(ss, token, '/')) {
            if (token.empty() || token == ".") {
                continue;
            } else if (token == "..") {
                if (!stack.empty()) {
                    stack.pop_back();
                }
            } else {
                stack.push_back(token);
            }
        }

        if (stack.empty()) {
            return "/";
        }

        std::string result = "";
        for (const std::string& dir : stack) {
            result += "/" + dir;
        }

        return result;
    }
};
```

#### Java

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class Solution {
    public String simplifyPath(String path) {
        Deque<String> stack = new ArrayDeque<>();
        String[] tokens = path.split("/");

        for (String token : tokens) {
            if (token.isEmpty() || token.equals(".")) {
                continue;
            } else if (token.equals("..")) {
                if (!stack.isEmpty()) {
                    stack.pop();
                }
            } else {
                stack.push(token);
            }
        }

        if (stack.isEmpty()) {
            return "/";
        }

        StringBuilder sb = new StringBuilder();
        // ArrayDeque iterates from top to bottom when using standard iteration,
        // so we iterate descendingly from bottom to top to preserve path order.
        var iterator = stack.descendingIterator();
        while (iterator.hasNext()) {
            sb.append("/").append(iterator.next());
        }

        return sb.toString();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$
  - Splitting the string and tokenizing takes $\mathcal{O}(N)$ where $N$ is the length of `path`.
  - Pushing and popping each token takes $\mathcal{O}(1)$ amortized.
  - Final string joining takes $\mathcal{O}(N)$ time.
  - Total Time: $\mathcal{O}(N)$.
- **Space Complexity:** $\mathcal{O}(N)$
  - The stack and tokens hold strings whose total length does not exceed $N$.

---

### Takeaway Pattern & Interview Traps

1. **Root Directory Underflow:**
   - Attempting to go up from root (`"/../"`) should simply stay at root rather than throwing an underflow error or pushing `".."`. Always verify `if stack:` before popping.
2. **Dots as Names:**
   - Only single dot `"."` and double dot `".."` have special meanings. Sequences like `"... "` or `"..hidden"` are valid file names and must be pushed normally.
3. **Empty String vs Root Slash:**
   - If the stack ends up empty (e.g. `path = "/a/.."`), the canonical path is `"/"`, not `""`.