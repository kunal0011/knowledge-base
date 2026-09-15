---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 179: Largest Number"
tags:
  - leetcode
  - coding
  - greedy
  - sorting
  - string
  - custom-comparator
  - amazon
  - google
---

# LeetCode 179: Largest Number

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Sorting / Custom Comparator / String  

---

### Problem Statement

Given a list of non-negative integers `nums`, arrange them such that they form the largest number and return it.

Since the result may be very large, so you need to return a string instead of an integer.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{nums.length} \le 100$).
- **Output:**
  - `str` / `string` — the concatenated largest number formatted as a string.
- **Constraints:**
  - $1 \le \text{nums.length} \le 100$
  - $0 \le \text{nums}[i] \le 10^9$

---

### Key Idea & Intuition

Normal numerical sorting fails (e.g. $30 > 3$, but concatenating $"3" + "30" = "330" > "303" = "30" + "3"$).
Lexicographical sorting also fails on prefixes (e.g. `"34"` vs `"3"` vs `"30"`).

#### The Greedy Concatenation Comparator:
For any two numbers $A$ and $B$ represented as strings, consider their relative ordering in the final concatenated string:
- If we place $A$ before $B$, their contribution is $A + B$.
- If we place $B$ before $A$, their contribution is $B + A$.

Therefore:
$$A \succ B \iff A + B > B + A$$

#### Proof of Transitivity (Total Order):
To use a comparison-based sort ($O(n \log n)$), the relation $\succ$ must define a **total preorder**:
1. **Reflexivity:** $A + A = A + A$, so $A \sim A$.
2. **Antisymmetry:** If $A + B = B + A$, $A$ and $B$ are equivalent in ranking.
3. **Transitivity:** If $A + B \ge B + A$ and $B + C \ge C + B$, then $A + C \ge C + A$.
   *(Proof: Let $l_A = 10^{|A|}$. Then $A + B = A \cdot l_B + B$. The condition $A \cdot l_B + B \ge B \cdot l_A + A \iff A(l_B - 1) \ge B(l_A - 1) \iff \frac{A}{l_A - 1} \ge \frac{B}{l_B - 1}$. Transitivity follows immediately from the ordering of real numbers).*

Since the comparison defines a strict weak ordering, standard sorting algorithms (Quicksort/Timsort) sort the strings in $\mathcal{O}(n \log n)$ comparisons.

#### Edge Case (Leading Zeros):
If the highest element after sorting is `"0"`, all elements must be `"0"` (e.g. `[0, 0]`). We must return `"0"` rather than `"00"`.

---

### Solution Approach (Step-by-Step)

1. Convert all numbers in `nums` to strings.
2. Sort the array of strings using the custom comparator:
   - $A$ comes before $B$ if $A + B > B + A$.
3. Check the first element of the sorted list:
   - If `nums_str[0] == "0"`, return `"0"`.
4. Concatenate all sorted strings: `"".join(nums_str)`.
5. Return the concatenated string.

---

### Visual Algorithm Walkthrough

For `nums = [3, 30, 34, 5, 9]`:

```
String representations: ["3", "30", "34", "5", "9"]

Pairwise comparisons:
  "9" vs "5":
    "95" > "59" -> "9" > "5"
  "5" vs "34":
    "534" > "345" -> "5" > "34"
  "34" vs "3":
    "343" > "334" -> "34" > "3"
  "3" vs "30":
    "330" > "303" -> "3" > "30"

Sorted Order:
  ["9", "5", "34", "3", "30"]

Concatenate:
  "9" + "5" + "34" + "3" + "30" = "9534330"

Result = "9534330"
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [10, 2]`
- **Comparison:** `"2" + "10" = "210" > "102" = "10" + "2"`
- **Output:** `"210"`

#### Example 2:
- **Input:** `nums = [3, 30, 34, 5, 9]`
- **Output:** `"9534330"`

#### Example 3 (All Zeros):
- **Input:** `nums = [0, 0]`
- **Tracing:** Concatenation `"00"`. Leading zero check returns `"0"`.
- **Output:** `"0"`

---

### Multi-Language Implementations

#### Python 3
```python
from functools import cmp_to_key
from typing import List

class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        # Convert all numbers to strings
        strs = [str(x) for x in nums]
        
        # Custom comparator: return negative if a should come before b
        def compare(a: str, b: str) -> int:
            if a + b > b + a:
                return -1
            elif a + b < b + a:
                return 1
            else:
                return 0
                
        strs.sort(key=cmp_to_key(compare))
        
        # Edge case: if largest number is "0", all elements are 0
        if strs[0] == "0":
            return "0"
            
        return "".join(strs)
```

#### C++17
```cpp
#include <vector>
#include <string>
#include <algorithm>

class Solution {
public:
    std::string largestNumber(std::vector<int>& nums) {
        std::vector<std::string> strs;
        strs.reserve(nums.size());
        for (int x : nums) {
            strs.push_back(std::to_string(x));
        }
        
        // Custom comparator: A comes before B if A + B > B + A
        std::sort(strs.begin(), strs.end(), [](const std::string& a, const std::string& b) {
            return a + b > b + a;
        });
        
        // Edge case: multiple zeros
        if (strs[0] == "0") {
            return "0";
        }
        
        std::string result;
        for (const auto& s : strs) {
            result += s;
        }
        
        return result;
    }
};
```

#### Java 17
```java
import java.util.Arrays;

class Solution {
    public String largestNumber(int[] nums) {
        String[] strs = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            strs[i] = String.valueOf(nums[i]);
        }
        
        // Sort using custom comparator (b + a compares to a + b for descending order)
        Arrays.sort(strs, (a, b) -> (b + a).compareTo(a + b));
        
        // Leading zero check
        if (strs[0].equals("0")) {
            return "0";
        }
        
        StringBuilder sb = new StringBuilder();
        for (String s : strs) {
            sb.append(s);
        }
        
        return sb.toString();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(k \cdot n \log n)$ where $k \le 10$ is the maximum number of digits per number
  - String conversion takes $\mathcal{O}(n \cdot k)$.
  - The sort does $\mathcal{O}(n \log n)$ comparisons. Each comparison concatenates two strings of length $\le 2k$ and compares them in $\mathcal{O}(k)$ time.
  - Concatenation takes $\mathcal{O}(n \cdot k)$.
  - Total time is $\mathcal{O}(k \cdot n \log n)$, well within 5ms.
- **Space Complexity:** $\mathcal{O}(n \cdot k)$ auxiliary space
  - To store string conversions and sorting buffer.

---

### Takeaway Pattern & Interview Traps

- **The Concatenation Comparator Principle:** When sorting elements whose order interacts under concatenation, define the comparator directly on the composite string: `a + b > b + a`.
- **The "00" Trap:** If input is `[0, 0, 0]`, naive string joining produces `"000"`, which fails LeetCode tests. Always check `if strs[0] == "0": return "0"`.