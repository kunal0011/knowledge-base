---
date: "2025-12-19"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 2: Add Two Numbers"
tags:
  - leetcode
  - coding
  - linked-list
---

# LeetCode 2: Add Two Numbers

Below is a complete, interview-ready explanation of **LeetCode 2 – Add Two Numbers**, structured exactly as requested.

---

## 1. Problem Statement

You are given **two non-empty linked lists** representing two **non-negative integers**.

* The digits are stored in **reverse order**
* Each node contains a **single digit**
* Add the two numbers and return the **sum as a linked list**

### Constraints

* The two numbers do not contain leading zero, except the number `0`
* Length of each list is in `[1, 100]`
* Each node value is in `[0, 9]`

---

### Example

```text
Input:
l1 = [2 → 4 → 3]   (represents 342)
l2 = [5 → 6 → 4]   (represents 465)

Output:
[7 → 0 → 8]        (represents 807)
```

---

## 2. Key Observation

1. **Digits are already reversed**, so:

   * Head = least significant digit
   * We can add digit-by-digit from head to tail
2. This mimics **manual addition**, including **carry**
3. Linked list traversal naturally aligns with addition order

---

## 3. Linked List Idea (Core Insight)

We simulate elementary school addition:

1. Traverse both lists simultaneously
2. At each step:

   ```
   sum = digit1 + digit2 + carry
   new_digit = sum % 10
   carry = sum // 10
   ```
3. Create a new node for `new_digit`
4. Continue until:

   * both lists are exhausted
   * and no carry remains

### Why use a dummy node?

* Simplifies handling of head
* Avoids special cases for first insertion

---

## 4. Python 3 Implementation (with Typing)

```python
from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(
        self,
        l1: Optional[ListNode],
        l2: Optional[ListNode]
    ) -> Optional[ListNode]:

        dummy = ListNode(0)
        current = dummy
        carry = 0

        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0

            total = val1 + val2 + carry
            carry = total // 10

            current.next = ListNode(total % 10)
            current = current.next

            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next

        return dummy.next
```

---

## 5. Worked-Out Example (Step-by-Step)

### Input

```
l1 = 2 → 4 → 3
l2 = 5 → 6 → 4
```

### Initialization

```
carry = 0
result = dummy → ∅
```

---

### Iteration 1

```
val1 = 2, val2 = 5
sum = 2 + 5 + 0 = 7
digit = 7, carry = 0
```

Result list:

```
7
```

---

### Iteration 2

```
val1 = 4, val2 = 6
sum = 4 + 6 + 0 = 10
digit = 0, carry = 1
```

Result list:

```
7 → 0
```

---

### Iteration 3

```
val1 = 3, val2 = 4
sum = 3 + 4 + 1 = 8
digit = 8, carry = 0
```

Result list:

```
7 → 0 → 8
```

---

### Final Output

```
[7 → 0 → 8]
```

---

## 6. Complexity Analysis

| Metric | Value |
| --- | --- |
| Time Complexity | **O(max(m, n))** |
| Space Complexity | **O(max(m, n))** (new list) |

---

## 7. Common Interview Pitfalls

* Forgetting the **final carry**
* Mishandling unequal list lengths
* Not using a dummy node (leads to messy head logic)

---

## 8. Pattern Recognition

This problem is a classic example of:

* **Linked List + Simulation**
* **Carry propagation**
* **Parallel traversal**

Once mastered, this pattern applies to:

* Multiplication of numbers in linked lists
* Binary linked list addition
* Big integer arithmetic problems

---

If you want, I can next:

* Show **recursive version**
* Extend to **forward-order digits**
* Convert this into a **template for all carry-based linked list problems**