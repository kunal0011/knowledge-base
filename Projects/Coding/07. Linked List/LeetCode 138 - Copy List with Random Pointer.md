---
date: "2026-08-29"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 138: Copy List with Random Pointer"
tags:
  - leetcode
  - coding
  - linked-list
---

# LeetCode 138: Copy List with Random Pointer

---

### Problem Statement

Construct a deep copy of a linked list where each node contains an additional `random` pointer pointing to any node in the list or null.

---

### Key Observation

* Approach 1: Hash Map `old_node -> new_node` in `O(n)` space.
* Approach 2: Interweave copied nodes directly after originals (`A -> A' -> B -> B'`) to achieve `O(1)` auxiliary space.
* Set random pointers: `curr.next.random = curr.random.next`, then decouple the lists.

---

### Core Technique: Interweaving Node Duplication (O(1) Space)

---

### Python 3 Solution (with typing)

```python
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return None
            
        # 1. Interweave copy nodes
        curr = head
        while curr:
            copy = Node(curr.val, curr.next)
            curr.next = copy
            curr = copy.next
            
        # 2. Assign random pointers
        curr = head
        while curr:
            if curr.random:
                curr.next.random = curr.random.next
            curr = curr.next.next
            
        # 3. Decouple lists
        curr = head
        copy_head = head.next
        while curr:
            copy = curr.next
            curr.next = copy.next
            curr = curr.next
            if copy.next:
                copy.next = copy.next.next
                
        return copy_head
```

---

### Worked-Out Example

```python
A -> B
Pass 1: A -> A' -> B -> B'
Pass 2: If A.random = B, then A'.random = A.random.next = B'
Pass 3: Separate into [A -> B] and [A' -> B']
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(1) auxiliary`

---

### Takeaway Pattern

Interweave copied nodes into original list to resolve arbitrary pointer graphs in O(1) space.