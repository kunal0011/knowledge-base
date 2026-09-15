---
date: "2026-09-15"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 206: Reverse Linked List"
tags:
  - leetcode
  - coding
  - linked-list
  - amazon
  - google
---

# LeetCode 206: Reverse Linked List

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Apple, Microsoft  
**Difficulty:** Easy  
**Topic:** In-Place Pointer Reversal / Iterative & Recursive

---

### Problem Statement

Given the `head` of a singly linked list, reverse the list, and return the reversed list.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head
        
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
            
        return prev
```

#### 2. C++ (C++17 / STL)
```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* curr = head;

        while (curr != nullptr) {
            ListNode* nxt = curr->next;
            curr->next = prev;
            prev = curr;
            curr = nxt;
        }
        return prev;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class ListNode {
    int val;
    ListNode next;
    ListNode(int val) { this.val = val; }
}

class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;

        while (curr != null) {
            ListNode nxt = curr.next;
            curr.next = prev;
            prev = curr;
            curr = nxt;
        }
        return prev;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ single pass.
- **Space Complexity:** $O(1)$ in-place iterative reversal.
