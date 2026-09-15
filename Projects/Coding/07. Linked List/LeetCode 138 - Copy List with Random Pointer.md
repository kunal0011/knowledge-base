---
date: "2026-09-15"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 138: Copy List with Random Pointer"
tags:
  - leetcode
  - coding
  - linked-list
  - deep-copy
  - hash-table
  - amazon
  - google
---

# LeetCode 138: Copy List with Random Pointer

**Target Companies:** Amazon (Top #1 Classic Linked List Question), Microsoft, Google, Meta, Bloomberg  
**Difficulty:** Medium  
**Topic:** Interweaving Node Duplication in $O(1)$ Auxiliary Space  

---

### Problem Statement

A linked list of length $n$ is given such that each node contains an additional random pointer, which could point to any node in the list, or `null`.

Construct a **deep copy** of the list. The deep copy should consist of exactly $n$ **brand new** nodes, where each new node has its value set to the value of its corresponding original node. Both the `next` and `random` pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.

Return the head of the copied linked list.

---

### Input & Output Formats & Constraints

- **Input:** `head: Optional[Node]`
- **Output:** `Optional[Node]` (head of deep-copied list)
- **Constraints:**
  - $0 \le n \le 1000$
  - $-10^4 \le \text{Node.val} \le 10^4$
  - `Node.random` is `null` or points to some node in the linked list.

---

### Key Idea & Intuition

#### Approach 1: Hash Map ($O(N)$ Time, $O(N)$ Space)
Maintain a dictionary `mapping = {old_node: Node(old_node.val)}`.
In a second pass, set `mapping[node].next = mapping[node.next]` and `mapping[node].random = mapping[node.random]`.
While intuitive, this consumes $O(N)$ extra hash table memory.

#### Approach 2: Optimal Interweaving ($O(N)$ Time, $O(1)$ Auxiliary Space)
We avoid the hash table by using the original list's `next` pointers to temporarily store the mapping:
1. **Pass 1 (Interweave):** For each original node `curr`, create a duplicate `copy` and insert it immediately after `curr`:
   $$A \to B \to C \implies A \to A' \to B \to B' \to C \to C'$$
2. **Pass 2 (Assign Randoms):** For every original node `curr`, its clone is at `curr.next`. If `curr.random` exists, the clone of `curr.random` is located at `curr.random.next`! Thus:
   $$\text{curr.next.random} = \text{curr.random.next}$$
3. **Pass 3 (Decouple):** Unweave the two intertwined lists, restoring the original list and extracting the cloned list.

---

### Solution Approach (Step-by-Step)

1. If `head` is null, return `None`.
2. **Pass 1:** Iterate `curr = head`. Create `copy = Node(curr.val, curr.next)`. Point `curr.next = copy`. Move `curr = copy.next`.
3. **Pass 2:** Iterate `curr = head`. While `curr`:
   - If `curr.random` is not null: `curr.next.random = curr.random.next`.
   - `curr = curr.next.next`.
4. **Pass 3:** Separate lists:
   - `curr = head`, `copy_head = head.next`.
   - While `curr`:
     - `copy = curr.next`.
     - `curr.next = copy.next`.
     - `copy.next = copy.next.next if copy.next else None`.
     - `curr = curr.next`.
5. Return `copy_head`.

---

### Visual Algorithm Walkthrough

```
Original: A -> B -> null
Pointers: A.random = B, B.random = A

Pass 1: Interweave copies
  A -> A' -> B -> B' -> null

Pass 2: Set random pointers
  A.random is B.
  A'.random = A.random.next = B'
  B.random is A.
  B'.random = B.random.next = A'

Pass 3: Decouple original and copied lists
  Original restored: A -> B -> null
  Copy extracted:    A' -> B' -> null (with A'.random = B', B'.random = A')

Return A' (copy_head).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Cross-Linked Random Pointers
- **Input:** `head = [[7,null],[13,0],[11,4],[10,2],[1,0]]`
- **Output:** Identical values and cloned structure with brand-new memory addresses.

#### Example 2: Self-Pointing Random
- **Input:** `head = [[1,1],[2,1]]`
- **Trace:** Node 1 points to itself via `random`. Its clone Node 1' correctly points to Node 1'.
- **Output:** `[[1,1],[2,1]]`

#### Example 3: Empty List
- **Input:** `head = []`
- **Output:** `null`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import Optional

class Node:
    def __init__(self, x: int, next: Optional['Node'] = None, random: Optional['Node'] = None):
        self.val = int(x)
        self.next = next
        self.random = random

class Solution:
    def copyRandomList(self, head: Optional[Node]) -> Optional[Node]:
        if not head:
            return None
            
        # 1. Interweave copy nodes
        curr = head
        while curr:
            copy = Node(curr.val, curr.next)
            curr.next = copy
            curr = copy.next
            
        # 2. Assign random pointers for copies
        curr = head
        while curr:
            if curr.random:
                curr.next.random = curr.random.next
            curr = curr.next.next
            
        # 3. Decouple the intertwined lists
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

#### 2. C++ (C++17 / STL)
```cpp
class Node {
public:
    int val;
    Node* next;
    Node* random;
    Node(int _val) : val(_val), next(nullptr), random(nullptr) {}
};

class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (!head) return nullptr;

        // 1. Interweave clone nodes
        Node* curr = head;
        while (curr != nullptr) {
            Node* copy = new Node(curr->val);
            copy->next = curr->next;
            curr->next = copy;
            curr = copy->next;
        }

        // 2. Assign random pointers
        curr = head;
        while (curr != nullptr) {
            if (curr->random != nullptr) {
                curr->next->random = curr->random->next;
            }
            curr = curr->next->next;
        }

        // 3. Decouple lists
        curr = head;
        Node* copyHead = head->next;
        while (curr != nullptr) {
            Node* copy = curr->next;
            curr->next = copy->next;
            curr = curr->next;
            if (copy->next != nullptr) {
                copy->next = copy->next->next;
            }
        }

        return copyHead;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Node {
    int val;
    Node next;
    Node random;
    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}

class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) return null;

        // 1. Interweave clone nodes
        Node curr = head;
        while (curr != null) {
            Node copy = new Node(curr.val);
            copy.next = curr.next;
            curr.next = copy;
            curr = copy.next;
        }

        // 2. Assign random pointers
        curr = head;
        while (curr != null) {
            if (curr.random != null) {
                curr.next.random = curr.random.next;
            }
            curr = curr.next.next;
        }

        // 3. Decouple lists
        curr = head;
        Node copyHead = head.next;
        while (curr != null) {
            Node copy = curr.next;
            curr.next = copy.next;
            curr = curr.next;
            if (copy.next != null) {
                copy.next = copy.next.next;
            }
        }

        return copyHead;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Exactly three linear passes over the linked list.
- **Space Complexity:** $O(1)$ auxiliary space — Beyond allocating the required $N$ new copied nodes, zero extra data structures (hash maps, arrays) are used.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Node Interleaving for Pointer Graph Cloning.
- **Trap:** Failing to restore the original linked list: Many candidates successfully build the copy list but leave the original list damaged (with broken `next` pointers). The unweaving pass must cleanly restore the original list's links.