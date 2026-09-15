---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 860: Lemonade Change"
tags:
  - leetcode
  - coding
  - greedy
  - array
  - simulation
  - amazon
  - google
  - bloomberg
---

# LeetCode 860: Lemonade Change

**Target Companies:** Amazon, Google, Bloomberg, Microsoft  
**Difficulty:** Easy  
**Topic:** Greedy / Array / Simulation  

---

### Problem Statement

At a lemonade stand, each lemonade costs `$5`. Customers are standing in a queue to buy from you and order one at a time (in the order specified by integer array `bills`). Each customer will only buy one lemonade and pay with either a `$5`, `$10`, or `$20` bill. You must provide the correct change to each customer so that the net transaction is that the customer pays `$5`.

Note that you begin with no change in hand.

Given an integer array `bills` where `bills[i]` is the bill the $i^{\text{th}}$ customer pays with, return `true` *if you can provide every customer with the correct change, or* `false` *otherwise*.

---

### Input & Output Formats & Constraints

- **Input:** An integer array `bills` ($1 \le |bills| \le 10^5$), where each $\text{bills}[i] \in \{5, 10, 20\}$.
- **Output:** A boolean (`true` or `false`).
- **Constraints:**
  - `1 <= bills.length <= 10^5`
  - `bills[i]` is either `5`, `10`, or `20`.

---

### Key Idea & Intuition

#### Bill Versatility & Change Invariant
Each customer requires exact change:
1. Customer pays **$5**: Change required is **$0**. Keep the $5 bill.
2. Customer pays **$10**: Change required is **$5**. Must provide one $5 bill.
3. Customer pays **$20**: Change required is **$15**.
   - Option A: One $10 bill + one $5 bill ($10 + $5 = $15$).
   - Option B: Three $5 bills ($5 + 5 + 5 = $15$).

#### Why Greedy Prioritization is Strictly Optimal
Notice the asymmetry in utility between $5 bills and $10 bills:
- A **$5 bill** can be used to make change for both a $10 bill and a $20 bill.
- A **$10 bill** can **only** be used to make change for a $20 bill (never for a $10 bill).
- A **$20 bill** can never be used to make change at all.

Therefore, the $5 bill is strictly more versatile than the $10 bill. When making $15 change for a $20 bill, we must **greedily prioritize spending the $10 bill first** (Option A). Conserving $5 bills maximizes our ability to satisfy upcoming $10 transactions.

---

### Solution Approach (Step-by-Step)

1. **State Tracking:**
   - Maintain two integer counters: `five = 0` and `ten = 0`.
2. **Process Each Bill in Sequence:**
   - If `bill == 5`:
     - Increment `five += 1`.
   - Else if `bill == 10`:
     - If `five == 0`, return `false` (cannot make change).
     - Decrement `five -= 1` and increment `ten += 1`.
   - Else (`bill == 20`):
     - If `ten > 0` and `five > 0`:
       - Use one $10 and one $5: `ten -= 1`, `five -= 1`.
     - Else if `five >= 3`:
       - Use three $5 bills: `five -= 3`.
     - Else:
       - Return `false` (insufficient bills to return $15).
3. **Completion:**
   - If all customers are served successfully, return `true`.

---

### Visual Algorithm Walkthrough

#### Trace for `bills = [5, 5, 5, 10, 20]`
```
Initial: five = 0, ten = 0

1. Customer pays 5:
   five = 1, ten = 0

2. Customer pays 5:
   five = 2, ten = 0

3. Customer pays 5:
   five = 3, ten = 0

4. Customer pays 10:
   Needs 5 change -> Give 1x $5 bill.
   five = 2, ten = 1

5. Customer pays 20:
   Needs 15 change -> Greedily prefer (1x $10 + 1x $5) over (3x $5).
   Give 1x $10 and 1x $5.
   five = 1, ten = 0

All customers served successfully! -> Return true.
```

#### Trace for `bills = [5, 5, 10, 10, 20]`
```
1. Customer pays 5: five = 1, ten = 0
2. Customer pays 5: five = 2, ten = 0
3. Customer pays 10: Needs 5 -> five = 1, ten = 1
4. Customer pays 10: Needs 5 -> five = 0, ten = 2
5. Customer pays 20: Needs 15 ->
   - Check Option A (10 + 5): We have ten = 2, but five = 0 -> Cannot!
   - Check Option B (5 + 5 + 5): We have five = 0 -> Cannot!
   Change cannot be provided -> Return false.
```

---

### Solved Examples with Multiple Inputs

| Input `bills` | Simulation Tracing (`five`, `ten`) | Final Result | Explanation |
|---|---|---|---|
| `[5, 5, 5, 10, 20]` | $5 \to (1,0); 5 \to (2,0); 5 \to (3,0); 10 \to (2,1); 20 \to (1,0)$ | `true` | Standard successful greedy change |
| `[5, 5, 10, 10, 20]` | $5 \to (1,0); 5 \to (2,0); 10 \to (1,1); 10 \to (0,2); 20 \to$ fails | `false` | Ran out of \$5 bills for \$20 change |
| `[10]` | Needs \$5 immediately; `five == 0` | `false` | First customer cannot receive change |
| `[5, 5, 5, 20]` | $5 \to (1,0); 5 \to (2,0); 5 \to (3,0); 20 \to (0,0)$ | `true` | Fallback to three \$5 bills works |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def lemonadeChange(self, bills: list[int]) -> bool:
        five: int = 0
        ten: int = 0
        
        for bill in bills:
            if bill == 5:
                five += 1
            elif bill == 10:
                if five == 0:
                    return False
                five -= 1
                ten += 1
            else:  # bill == 20
                # Greedily give one $10 and one $5 to preserve $5 bills
                if ten > 0 and five > 0:
                    ten -= 1
                    five -= 1
                elif five >= 3:
                    five -= 3
                else:
                    return False
                    
        return True
```

#### C++17
```cpp
#include <vector>

class Solution {
public:
    bool lemonadeChange(const std::vector<int>& bills) {
        int five = 0;
        int ten = 0;
        
        for (int bill : bills) {
            if (bill == 5) {
                five++;
            } else if (bill == 10) {
                if (five == 0) return false;
                five--;
                ten++;
            } else { // bill == 20
                // Greedily use 10 + 5 first
                if (ten > 0 && five > 0) {
                    ten--;
                    five--;
                } else if (five >= 3) {
                    five -= 3;
                } else {
                    return false;
                }
            }
        }
        
        return true;
    }
};
```

#### Java 17
```java
class Solution {
    public boolean lemonadeChange(int[] bills) {
        int five = 0;
        int ten = 0;
        
        for (int bill : bills) {
            if (bill == 5) {
                five++;
            } else if (bill == 10) {
                if (five == 0) {
                    return false;
                }
                five--;
                ten++;
            } else { // bill == 20
                // Prioritize spending larger denomination $10 bill
                if (ten > 0 && five > 0) {
                    ten--;
                    five--;
                } else if (five >= 3) {
                    five -= 3;
                } else {
                    return false;
                }
            }
        }
        
        return true;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$, where $N = |bills|$. We traverse the array once, performing $\mathcal{O}(1)$ arithmetic and condition checks per transaction.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space, as only two scalar integer counters (`five`, `ten`) are maintained.

---

### Takeaway Pattern & Interview Traps

1. **Why No $20 Counter?** A $20 bill can never be given as change because the maximum change for any transaction is $15. Tracking $20 bills is redundant.
2. **Greedy Dominance:** Giving three $5 bills instead of one $10 and one $5 when both are available is suboptimal because saving the $5 bill protects against future $10 customers.
3. **Queue Order:** Customers must be served strictly in sequential order; sorting the queue or reordering is illegal per problem specifications.