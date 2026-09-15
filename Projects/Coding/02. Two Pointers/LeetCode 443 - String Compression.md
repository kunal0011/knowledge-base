---
date: "2025-12-22"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 443: String Compression"
tags:
  - leetcode
  - coding
  - two-pointers
---

# LeetCode 443: String Compression

Below is a complete, interview-ready explanation of **LeetCode 443 – String Compression**, structured exactly as requested.

---

## LeetCode 443 – String Compression

### Problem Statement

You are given an array of characters `chars`.  
Compress the array **in-place** using the following rules:

* For a group of **consecutive repeating characters**:

  * Write the character once.
  * If the count is greater than `1`, append the **digits of the count**.
* The compressed result must be stored in the **original array**.
* Return the **new length** of the array after compression.

**Constraints**

* Do not allocate extra space for another array.
* The input array length can be large.

---

### Key Observation

1. Compression depends only on **consecutive runs** of characters.
2. We must:

   * **Read** the original characters sequentially.
   * **Write** the compressed output back into the same array.
3. Therefore, **two pointers** are required:

   * One pointer to **scan** the input.
   * One pointer to **write** the compressed result.

This is a classic **read–write in-place transformation** problem.

---

### Two Pointer Technique (Core Insight)

We use:

* `read` → scans the input
* `write` → writes the compressed output

#### High-level Flow

1. Start `read = 0`
2. While `read < n`:

   * Count how many times `chars[read]` repeats.
   * Write the character to `chars[write]`
   * If count > 1, write each digit of the count
3. Move `read` to the next new character
4. Return `write` as the compressed length

This ensures:

* **O(n)** time complexity
* **O(1)** extra space

---

### Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def compress(self, chars: List[str]) -> int:
        write = 0
        read = 0
        n = len(chars)

        while read < n:
            char = chars[read]
            count = 0

            # Count consecutive characters
            while read < n and chars[read] == char:
                read += 1
                count += 1

            # Write the character
            chars[write] = char
            write += 1

            # Write the count if > 1
            if count > 1:
                for digit in str(count):
                    chars[write] = digit
                    write += 1

        return write
```

---

### Worked Out Example

#### Input

```
chars = ["a","a","b","b","c","c","c"]
```

---

#### Step-by-Step Execution

| Read Group | Count | Write Operations | chars after write |
| --- | --- | --- | --- |
| `'a'` | 2 | write `'a'`, `'2'` | `["a","2",...]` |
| `'b'` | 2 | write `'b'`, `'2'` | `["a","2","b","2",...]` |
| `'c'` | 3 | write `'c'`, `'3'` | `["a","2","b","2","c","3"]` |

---

#### Final Output

```
Compressed chars = ["a","2","b","2","c","3"]
Return length = 6
```

---

### Edge Case Examples

1. **Single character**

   ```text
   Input: ["a"]
   Output: ["a"], length = 1
   ```
2. **Count ≥ 10**

   ```text
   Input: ["a","a","a","a","a","a","a","a","a","a","a","a"]
   Output: ["a","1","2"], length = 3
   ```

   Note: Count is split into digits.

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(1)` (in-place)

---

### Interview Takeaway

* This is a **two-pointer + in-place write** pattern.
* Always separate **reading** from **writing**.
* Whenever counts can exceed single digits, convert count to string and write digit-by-digit.

If you want, I can also provide:

* A dry-run visualization with pointer movement
* Common pitfalls and why naive solutions fail
* Similar pattern problems to practice next