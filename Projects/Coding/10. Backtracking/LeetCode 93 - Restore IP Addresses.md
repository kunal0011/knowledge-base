---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 93: Restore IP Addresses"
tags:
  - leetcode
  - coding
  - backtracking
  - string
  - amazon
  - google
---

# LeetCode 93: Restore IP Addresses

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Backtracking / String Partitioning / IP Octet Validation  

---

### Problem Statement

A **valid IP address** consists of exactly four integers separated by single dots. Each integer is between `0` and `255` (inclusive) and cannot have leading zeros.

- For example, `"0.1.2.201"` and `"192.168.1.1"` are valid IP addresses, but `"0.011.255.245"`, `"192.168.1.312"` and `"192.168@1.1"` are invalid IP addresses.

Given a string `s` containing only digits, return *all possible valid IP addresses that can be formed by inserting dots into `s`*. You are not allowed to reorder or remove any digits in `s`. You may return the valid IP addresses in **any order**.

---

### Input & Output Formats & Constraints

- **Input:** `s: str` (contains only digits `'0'`-`'9'`)
- **Output:** `List[str]` containing formatted 4-octet IP addresses.
- **Constraints:**
  - $1 \le \text{s.length} \le 20$
  - `s` consists of digits only.

---

### Key Idea & Intuition

- **Boundary Invariants on Length:**
  - An IPv4 address has exactly 4 octets.
  - Each octet has between 1 and 3 digits.
  - Therefore, any valid input string must have length between:
    $$4 \le \text{s.length} \le 12$$
  - If $\text{len}(s) < 4$ or $\text{len}(s) > 12$, immediately return `[]`.
- **Octet Validation Rules:**
  - Given a candidate substring `seg = s[start : start + length]`:
    1. **Length:** $1 \le \text{length} \le 3$.
    2. **No Leading Zeros:** If $\text{len}(seg) > 1$ and $seg[0] == \text{'0'}$, the octet is invalid (e.g., `"01"` is invalid, but `"0"` is valid).
    3. **Value Range:** $0 \le \text{int}(seg) \le 255$.
- **Aggressive Remaining-Length Pruning:**
  - If `len(segments) == 4` and `start == len(s)`: record the full IP address.
  - At any point, if the remaining characters cannot be partitioned into the remaining octets:
    $$\text{remaining\_chars} < (4 - \text{len(segments)}) \times 1 \quad \lor \quad \text{remaining\_chars} > (4 - \text{len(segments)}) \times 3$$
    we can immediately prune that branch.

---

### Solution Approach (Step-by-Step)

1. Check upfront: if $\text{len}(s) < 4$ or $\text{len}(s) > 12$, return `[]`.
2. Initialize `results = []` and `segments = []`.
3. Define `backtrack(start)`:
   - If `len(segments) == 4`:
     - If `start == len(s)`:
       - Append `'.'.join(segments)` to `results`.
     - Return.
   - For `length` in `(1, 2, 3)`:
     - If `start + length > len(s)`:
       - Break.
     - `seg = s[start : start + length]`
     - **Check Leading Zero:** If `length > 1 and seg[0] == '0'`:
       - Break (cannot form valid 2- or 3-digit octet starting with '0').
     - **Check Numeric Value:** If `int(seg) > 255`:
       - Break (any longer slice will also exceed 255).
     - **Choose:** `segments.append(seg)`
     - **Explore:** `backtrack(start + length)`
     - **Backtrack:** `segments.pop()`
4. Call `backtrack(0)` and return `results`.

---

### Visual Algorithm Walkthrough

Let `s = "25525511135"`:

```
                            backtrack(start=0, segments=[])
                      /                    |                 \
                "2"                     "25"                "255"
               /                         |                    |
             ...                        ...             segments=["255"]
                                                        start=3, rem="25511135"
                                                              |
                                                     "255" (valid, <= 255)
                                                        segments=["255", "255"]
                                                        start=6, rem="11135"
                                                        /                  \
                                                  "11" (len 2)          "111" (len 3)
                                                 start=8, rem="135"    start=9, rem="35"
                                                       |                     |
                                                  "135" (len 3)         "35" (len 2)
                                                 start=11 (end!)       start=11 (end!)
                                                       |                     |
                                              "255.255.11.135"       "255.255.111.35"
```

Both paths successfully reached 4 valid octets using all 11 characters.

---

### Solved Examples with Multiple Inputs

| Test Case | `s` | Valid Segments Partitioning | Output |
| :--- | :--- | :--- | :--- |
| **Standard** | `"25525511135"` | `255.255.11.135` and `255.255.111.35` | `["255.255.11.135","255.255.111.35"]` |
| **All Zeros** | `"0000"` | Exactly `0.0.0.0` | `["0.0.0.0"]` |
| **Leading Zeros** | `"101023"` | Cannot have `"02"`, `"023"` | `["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]` |
| **Too Short** | `"111"` | $< 4$ characters | `[]` |
| **Too Long** | `"2552552552551"` | $> 12$ characters | `[]` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        """
        Restores all possible valid IPv4 addresses from string s.
        Uses 4-octet bounded backtracking with strict octet validation.
        """
        n = len(s)
        if n < 4 or n > 12:
            return []

        results: List[str] = []
        segments: List[str] = []

        def backtrack(start: int) -> None:
            # If 4 octets have been placed
            if len(segments) == 4:
                if start == n:
                    results.append(".".join(segments))
                return

            # Pruning: remaining characters must fit in remaining octets
            remaining_octets = 4 - len(segments)
            remaining_chars = n - start
            if remaining_chars < remaining_octets or remaining_chars > remaining_octets * 3:
                return

            for length in (1, 2, 3):
                if start + length > n:
                    break

                seg = s[start : start + length]

                # Check leading zero: '0' alone is valid, '0X' is invalid
                if length > 1 and seg[0] == '0':
                    break

                # Check numeric value <= 255
                if int(seg) > 255:
                    break

                segments.append(seg)
                backtrack(start + length)
                segments.pop()  # Backtrack

        backtrack(0)
        return results
```

#### C++17
```cpp
#include <string>
#include <vector>

class Solution {
public:
    std::vector<std::string> restoreIpAddresses(const std::string& s) {
        int n = static_cast<int>(s.size());
        if (n < 4 || n > 12) return {};

        std::vector<std::string> results;
        std::vector<std::string> segments;
        backtrack(0, n, s, segments, results);
        return results;
    }

private:
    void backtrack(int start, int n, const std::string& s,
                   std::vector<std::string>& segments,
                   std::vector<std::string>& results) {
        if (segments.size() == 4) {
            if (start == n) {
                results.push_back(segments[0] + "." + segments[1] + "." + segments[2] + "." + segments[3]);
            }
            return;
        }

        int remaining_octets = 4 - static_cast<int>(segments.size());
        int remaining_chars = n - start;
        if (remaining_chars < remaining_octets || remaining_chars > remaining_octets * 3) {
            return;
        }

        for (int len = 1; len <= 3 && start + len <= n; ++len) {
            std::string seg = s.substr(start, len);

            // Leading zero check
            if (len > 1 && seg[0] == '0') break;

            // Range check
            int val = std::stoi(seg);
            if (val > 255) break;

            segments.push_back(seg);
            backtrack(start + len, n, s, segments, results);
            segments.pop_back(); // Backtrack
        }
    }
};
```

#### Java
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<String> restoreIpAddresses(String s) {
        int n = s.length();
        List<String> results = new ArrayList<>();
        if (n < 4 || n > 12) {
            return results;
        }

        List<String> segments = new ArrayList<>();
        backtrack(0, n, s, segments, results);
        return results;
    }

    private void backtrack(int start, int n, String s, List<String> segments, List<String> results) {
        if (segments.size() == 4) {
            if (start == n) {
                results.add(String.join(".", segments));
            }
            return;
        }

        int remainingOctets = 4 - segments.size();
        int remainingChars = n - start;
        if (remainingChars < remainingOctets || remainingChars > remainingOctets * 3) {
            return;
        }

        for (int len = 1; len <= 3 && start + len <= n; len++) {
            String seg = s.substring(start, start + len);

            // Check leading zero
            if (len > 1 && seg.charAt(0) == '0') {
                break;
            }

            // Check integer value <= 255
            int val = Integer.parseInt(seg);
            if (val > 255) {
                break;
            }

            segments.add(seg);
            backtrack(start + len, n, s, segments, results);
            segments.remove(segments.size() - 1); // Backtrack
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(1)$ strictly constant upper bound.
  - Tree depth is fixed at 4 octets.
  - Each octet has at most 3 choices ($1, 2, 3$ digits).
  - Total states visited $\le 3^4 = 81$ states!
  - Formatting takes $\mathcal{O}(1)$ time. Executes in $< 1 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space for the recursion stack (depth at most 4) and list of 4 segments.

---

### Takeaway Pattern & Interview Traps

- **Length Constraints as Upfront Guards:** Checking $4 \le \text{len}(s) \le 12$ immediately eliminates strings that cannot possibly form an IPv4 address.
- **`break` vs `continue` on Leading Zeros:** If `seg[0] == '0'` and `length > 1`, you must `break`, not `continue`! A segment starting with `'0'` can never become valid with additional digits (`"05"` and `"053"` are both invalid).