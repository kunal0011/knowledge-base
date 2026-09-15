---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 155: Min Stack (Single Stack Technique)"
tags:
  - leetcode
  - coding
  - stack
---

# LeetCode 155: Min Stack (Single Stack Technique)

Below is the **single-stack (value encoding) approach** for **LeetCode 155 – Min Stack**, structured in the same rigorous format.

---

## LeetCode 155 – Min Stack (Single Stack Technique)

### Core Idea

Instead of using a separate `min_stack`, we **encode values** in the main stack such that:

* We can detect when the **minimum changes**
* We can **restore the previous minimum** during `pop()`
* All operations remain **O(1)**

This approach relies on **mathematical encoding**, not an auxiliary stack.

---

## Key Observation

When a new value becomes the **new minimum**, we must:

1. Remember the **old minimum**
2. Push something that allows us to recover it later

We do this by pushing a **transformed value**.

---

## Stack Key Insight (Encoding Trick)

Maintain:

* One stack: `stack`
* One variable: `current_min`

### Encoding Rule

When pushing a value `x`:

#### Case 1: Stack is empty

```
push(x)
current_min = x
```

#### Case 2: x ≥ current\_min

```
push(x)
```

#### Case 3: x < current\_min (new minimum)

```
encoded_value = 2*x - current_min
push(encoded_value)
current_min = x
```

👉 The encoded value is **always smaller than the new minimum**, which lets us detect it later.

---

## Decoding During Pop

When popping:

* If `top ≥ current_min` → normal value
* If `top < current_min` → encoded value

  * Previous minimum is recovered using:

    ```
    previous_min = 2*current_min - encoded_value
    ```

---

## Python 3 Solution (with typing)

```python
from typing import List

class MinStack:
    def __init__(self) -> None:
        self.stack: List[int] = []
        self.current_min: int | None = None

    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append(val)
            self.current_min = val
        elif val >= self.current_min:
            self.stack.append(val)
        else:
            encoded = 2 * val - self.current_min
            self.stack.append(encoded)
            self.current_min = val

    def pop(self) -> None:
        top = self.stack.pop()
        if top < self.current_min:
            # Decode previous minimum
            self.current_min = 2 * self.current_min - top

        if not self.stack:
            self.current_min = None

    def top(self) -> int:
        top = self.stack[-1]
        if top < self.current_min:
            return self.current_min
        return top

    def getMin(self) -> int:
        return self.current_min
```

---

## Worked Out Example (Step-by-Step)

### Operations

```
push(5)
push(3)
push(7)
push(2)
top()
getMin()
pop()
getMin()
```

---

### Step 1: `push(5)`

| Stack | current\_min |
| --- | --- |
| [5] | 5 |

---

### Step 2: `push(3)` → New minimum

```
encoded = 2*3 - 5 = 1
```

| Stack | current\_min |
| --- | --- |
| [5, 1] | 3 |

---

### Step 3: `push(7)` (normal push)

| Stack | current\_min |
| --- | --- |
| [5, 1, 7] | 3 |

---

### Step 4: `push(2)` → New minimum

```
encoded = 2*2 - 3 = 1
```

| Stack | current\_min |
| --- | --- |
| [5, 1, 7, 1] | 2 |

---

### Step 5: `top()`

Top value = `1`  
Since `1 < current_min (2)` → encoded  
Return `current_min`

```
top() → 2
```

---

### Step 6: `getMin()`

```
current_min → 2
```

---

### Step 7: `pop()`

Pop `1` (encoded)

```
previous_min = 2*2 - 1 = 3
```

| Stack | current\_min |
| --- | --- |
| [5, 1, 7] | 3 |

---

### Step 8: `getMin()`

```
current_min → 3
```

---

## Why This Works

* Encoded values act as **markers** indicating a minimum change
* Mathematical transformation guarantees:

  ```
  encoded < new_min
  ```
* Old minimum can be reconstructed using the inverse formula

---

## Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time per operation | O(1) |
| Extra space | O(1) |

---

## Interview Comparison

| Approach | Extra Space | Complexity | Readability |
| --- | --- | --- | --- |
| Two Stack | O(n) | O(1) | Very High |
| Single Stack (Encoded) | O(1) | O(1) | Medium |

**Interview Tip:**  
Explain the **two-stack approach first**, then mention this as an **optimization**.

If you want, I can also provide a **formal invariant proof** or explain **why overflow is not an issue in Python but matters in Java/C++**.