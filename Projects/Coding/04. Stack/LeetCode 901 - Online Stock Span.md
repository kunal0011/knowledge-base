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
---

# LeetCode 901: Online Stock Span

Below is a complete, structured explanation of **LeetCode 901 – Online Stock Span**, aligned with interview expectations and stack-based reasoning.

---

## LeetCode 901: Online Stock Span

### Problem Statement

Design a class `StockSpanner` that collects daily stock prices and returns the **span** of the stock’s price for the current day.

The **span** of today’s price is defined as the maximum number of **consecutive days (ending today)** for which the price of the stock was **less than or equal to today’s price**.

You will receive prices **one by one**, and for each price you must return the span.

**Function signature**

```python
StockSpanner()
next(price: int) -> int
```

---

### Key Observation

For each day, we want to look **backwards** and count how many consecutive previous days have prices **≤ today’s price**.

A naïve approach would scan backwards for every call → **O(n)** per operation → **O(n²)** total.

We need a way to:

* Skip over smaller prices efficiently
* Avoid reprocessing the same prices multiple times

This naturally leads to a **monotonic stack**.

---

### Stack Key Insight (Monotonic Decreasing Stack)

Maintain a stack of pairs:

```
(price, span)
```

#### Stack invariant

* Prices in the stack are **strictly decreasing** from bottom to top.
* Each element represents a **compressed block** of days.

#### Why store span in the stack?

When a new price arrives:

* If it is **greater than or equal** to the top price,

  * That previous price (and its span) can be **merged** into today’s span.
* This avoids recounting past days.

Each price is:

* **Pushed once**
* **Popped once**

Hence total complexity is linear.

---

### Algorithm

For `next(price)`:

1. Initialize `span = 1` (today counts).
2. While stack is not empty and  
   `stack.top.price <= price`:

   * Pop `(prev_price, prev_span)`
   * Add `prev_span` to `span`
3. Push `(price, span)` onto stack.
4. Return `span`.

---

### Python 3 Solution (with typing)

```python
from typing import List, Tuple

class StockSpanner:
    def __init__(self) -> None:
        # Stack stores tuples of (price, span)
        self.stack: List[Tuple[int, int]] = []

    def next(self, price: int) -> int:
        span = 1

        # Merge spans of all previous prices <= current price
        while self.stack and self.stack[-1][0] <= price:
            _, prev_span = self.stack.pop()
            span += prev_span

        self.stack.append((price, span))
        return span
```

---

### Worked-Out Example

**Input sequence**

```
Prices = [100, 80, 60, 70, 60, 75, 85]
```

---

#### Day 1: price = 100

* Stack empty
* span = 1
* Push (100, 1)

Stack:

```
[(100, 1)]
```

Output: **1**

---

#### Day 2: price = 80

* 80 < 100 → no pop
* span = 1
* Push (80, 1)

Stack:

```
[(100, 1), (80, 1)]
```

Output: **1**

---

#### Day 3: price = 60

* 60 < 80 → no pop
* span = 1
* Push (60, 1)

Stack:

```
[(100, 1), (80, 1), (60, 1)]
```

Output: **1**

---

#### Day 4: price = 70

* 70 ≥ 60 → pop (60, 1), span = 2
* 70 < 80 → stop
* Push (70, 2)

Stack:

```
[(100, 1), (80, 1), (70, 2)]
```

Output: **2**

---

#### Day 5: price = 60

* 60 < 70 → no pop
* span = 1
* Push (60, 1)

Stack:

```
[(100, 1), (80, 1), (70, 2), (60, 1)]
```

Output: **1**

---

#### Day 6: price = 75

* 75 ≥ 60 → pop (60, 1), span = 2
* 75 ≥ 70 → pop (70, 2), span = 4
* 75 < 80 → stop
* Push (75, 4)

Stack:

```
[(100, 1), (80, 1), (75, 4)]
```

Output: **4**

---

#### Day 7: price = 85

* 85 ≥ 75 → pop (75, 4), span = 5
* 85 ≥ 80 → pop (80, 1), span = 6
* 85 < 100 → stop
* Push (85, 6)

Stack:

```
[(100, 1), (85, 6)]
```

Output: **6**

---

### Final Output

```
[1, 1, 1, 2, 1, 4, 6]
```

---

### Complexity Analysis

* **Time Complexity**:  
  Amortized **O(1)** per `next()` call  
  Overall **O(n)** for n prices
* **Space Complexity**:  
  **O(n)** stack in the worst case

---

### Interview Takeaway

* This is a **classic monotonic stack + span compression** problem.
* Key realization:  
  *Each element summarizes multiple days → avoid rescanning.*
* Same pattern appears in:

  * Next Greater Element
  * Histogram Largest Rectangle
  * Daily Temperatures

If you want, I can also:

* Draw the **monotonic stack evolution visually**
* Compare this with a **brute-force solution**
* Show how this maps to other monotonic stack problems