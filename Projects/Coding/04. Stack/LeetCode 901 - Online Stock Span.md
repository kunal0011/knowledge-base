---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 901: Online Stock Span"
tags:
  - leetcode
  - coding
  - stack
  - monotonic-stack
  - design
  - amazon
  - google
---

# LeetCode 901: Online Stock Span

**Target Companies:** Amazon, Google, Microsoft, Bloomberg, Meta  
**Difficulty:** Medium  
**Topic:** Monotonic Decreasing Stack / Online Stream / Span Compression

---

### Problem Statement

Design an algorithm that collects daily price quotes for some stock and returns the **span** of that stock's price for the current day.

The **span** of the stock's price in one day is the maximum number of consecutive days (starting from that day and going backward) for which the stock price was less than or equal to the price of that day.

- For example, if the prices of the stock in the last four days is `[7, 2, 1, 2]` and the price of the stock today is `2`, then the span of today is `4` because starting from today, the price of the stock was less than or equal to `2` for `4` consecutive days.
- Also, if the prices of the stock in the last four days is `[7, 34, 1, 2]` and the price of the stock today is `8`, then the span of today is `3` because starting from today, the price of the stock was less than or equal to `8` for `3` consecutive days.

Implement the `StockSpanner` class:
- `StockSpanner()`: Initializes the object of the class.
- `int next(int price)`: Returns the span of the stock's price given that today's price is `price`.

---

### Input & Output Formats & Constraints

- **Input:**
  - Method calls: `["StockSpanner", "next", "next", ...]` with arguments `[[price], ...]`.
- **Output:**
  - `int`: The span for the current day's price.
- **Constraints:**
  - $1 \le price \le 10^5$.
  - At most $10^4$ calls will be made to `next`.

---

### Key Idea & Intuition

A brute force approach would record all past prices in a list and scan backward on each call to `next(price)`, which takes $\mathcal{O}(N)$ per query, resulting in $\mathcal{O}(N^2)$ overall time.

#### Monotonic Decreasing Stack with Span Compression:
Notice that if today's price is higher than or equal to yesterday's price, today's price "dominates" yesterday's price for all future queries. Any future day that is $\ge$ today's price will automatically be $\ge$ yesterday's price as well!

Therefore, we can **compress** consecutive dominated days:
- Maintain a monotonic stack storing pairs: `(price, span)`.
- Stack invariant: prices in the stack are strictly decreasing from bottom to top.
- When a new price arrives:
  - Initialize `span = 1` (today itself counts as 1 day).
  - While `stack` is non-empty and `stack.top().price <= price`:
    - Pop `(prev_price, prev_span)`.
    - Accumulate `span += prev_span`.
  - Push `(price, span)` onto the stack.
  - Return `span`.

Every price is pushed onto the stack exactly once and popped at most once across the entire lifecycle of the object, guaranteeing **$\mathcal{O}(1)$ amortized time per operation**.

---

### Solution Approach (Step-by-Step)

1. **Class Initialization (`__init__`):**
   - Initialize `self.stack = []` (holding pairs of `[price, span]`).
2. **`next(price)`:**
   - Set `span = 1`.
   - While `stack` is non-empty and `stack[-1][0] <= price`:
     - Pop `(_, prev_span) = stack.pop()`.
     - `span += prev_span`.
   - Push `(price, span)` onto `stack`.
   - Return `span`.

---

### Visual Algorithm Walkthrough

Stream of prices: `[100, 80, 60, 70, 60, 75, 85]`

```
1. next(100):
   stack empty -> span = 1.
   Push (100, 1). Stack: [(100, 1)]
   Return 1

2. next(80):
   80 < 100 -> stack top is greater -> span = 1.
   Push (80, 1). Stack: [(100, 1), (80, 1)]
   Return 1

3. next(60):
   60 < 80 -> span = 1.
   Push (60, 1). Stack: [(100, 1), (80, 1), (60, 1)]
   Return 1

4. next(70):
   70 >= 60 -> Pop (60, 1), span becomes 1 + 1 = 2.
   70 < 80  -> Stop popping.
   Push (70, 2). Stack: [(100, 1), (80, 1), (70, 2)]
   Return 2

5. next(60):
   60 < 70 -> span = 1.
   Push (60, 1). Stack: [(100, 1), (80, 1), (70, 2), (60, 1)]
   Return 1

6. next(75):
   75 >= 60 -> Pop (60, 1), span = 1 + 1 = 2.
   75 >= 70 -> Pop (70, 2), span = 2 + 2 = 4! (Compressed past 4 days!)
   75 < 80  -> Stop popping.
   Push (75, 4). Stack: [(100, 1), (80, 1), (75, 4)]
   Return 4

7. next(85):
   85 >= 75 -> Pop (75, 4), span = 1 + 4 = 5.
   85 >= 80 -> Pop (80, 1), span = 5 + 1 = 6.
   85 < 100 -> Stop popping.
   Push (85, 6). Stack: [(100, 1), (85, 6)]
   Return 6

Outputs: [1, 1, 1, 2, 1, 4, 6]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Mixed Sequence

- **Calls:** `StockSpanner()`, followed by `next` on `[100, 80, 60, 70, 60, 75, 85]`
- **Output:** `[null, 1, 1, 1, 2, 1, 4, 6]`

#### Example 2: Monotonically Strictly Increasing

- **Calls:** `next` on `[10, 20, 30, 40]`
- **Tracing:**
  - `10` $\implies 1$
  - `20` $\implies$ pops `10` $\implies 1 + 1 = 2$
  - `30` $\implies$ pops `20` $\implies 1 + 2 = 3$
  - `40` $\implies$ pops `30` $\implies 1 + 3 = 4$
- **Output:** `[1, 2, 3, 4]`

#### Example 3: Monotonically Strictly Decreasing

- **Calls:** `next` on `[40, 30, 20, 10]`
- **Tracing:** Each price is smaller than the previous; no elements are popped.
- **Output:** `[1, 1, 1, 1]`

---

### Multi-Language Implementations

#### Python 3

```python
from typing import List, Tuple

class StockSpanner:
    def __init__(self):
        # Stack stores tuples of (price, span)
        self.stack: List[Tuple[int, int]] = []

    def next(self, price: int) -> int:
        span = 1
        # Merge spans of all consecutive preceding prices <= current price
        while self.stack and self.stack[-1][0] <= price:
            _, prev_span = self.stack.pop()
            span += prev_span

        self.stack.append((price, span))
        return span
```

#### C++17

```cpp
#include <vector>

class StockSpanner {
private:
    struct Element {
        int price;
        int span;
    };
    std::vector<Element> stack;

public:
    StockSpanner() {}

    int next(int price) {
        int span = 1;
        while (!stack.empty() && stack.back().price <= price) {
            span += stack.back().span;
            stack.pop_back();
        }
        stack.push_back({price, span});
        return span;
    }
};
```

#### Java

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class StockSpanner {
    private static class Node {
        int price;
        int span;

        Node(int price, int span) {
            this.price = price;
            this.span = span;
        }
    }

    private final Deque<Node> stack;

    public StockSpanner() {
        this.stack = new ArrayDeque<>();
    }

    public int next(int price) {
        int span = 1;
        while (!stack.isEmpty() && stack.peek().price <= price) {
            span += stack.pop().span;
        }
        stack.push(new Node(price, span));
        return span;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - Amortized $\mathcal{O}(1)$ time per `next(price)` invocation.
  - Across $N$ calls to `next`, each price is pushed onto the stack once and popped at most once. Total time for $N$ operations is $\mathcal{O}(N)$, giving an amortized cost of $\mathcal{O}(1)$ per call.
- **Space Complexity:** $\mathcal{O}(N)$
  - In the worst case (monotonically decreasing prices), the stack stores all $N$ prices.
  - Auxiliary space: $\mathcal{O}(N)$.

---

### Takeaway Pattern & Interview Traps

1. **Span Compression vs Index Storing:**
   - In an offline array setting (like LC 739 Daily Temperatures), storing indices `i` allows calculating distance via `i - prev_idx`.
   - In an **online stream** setting, storing the pre-aggregated `span` directly in the stack tuple `(price, span)` avoids maintaining a global counter or growing a history array, which allows infinite streaming without memory leaks for non-monotone data.
2. **Handling Non-Strict Inequality ($\le$):**
   - The problem specifies that prices *less than or equal to* today's price are included in the span. Ensure the comparison is `stack.top().price <= price`, not strictly `<`.