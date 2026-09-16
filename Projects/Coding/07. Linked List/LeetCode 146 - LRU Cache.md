---
date: "2026-09-15"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 146: LRU Cache"
tags:
  - leetcode
  - coding
  - linked-list
  - amazon
  - google
---

# LeetCode 146: LRU Cache

**Target Companies:** Amazon (Top #1 System/LLD Question), Google, Meta, Apple, Microsoft  
**Difficulty:** Medium  
**Topic:** Doubly Linked List + Hash Map (O(1) Get & Put)

---

### Problem Statement

Design a data structure that follows the constraints of a **Least Recently Used (LRU) cache**.

Implement the `LRUCache` class:
- `LRUCache(int capacity)`: Initialize the LRU cache with positive size `capacity`.
- `int get(int key)`: Return the value of the `key` if the `key` exists, otherwise return `-1`.
- `void put(int key, int value)`: Update the value of the `key` if the `key` exists. Otherwise, add the `key-value` pair to the cache. If the number of keys exceeds the `capacity` from this operation, **evict** the least recently used key.

The functions `get` and `put` must each run in **$O(1)$ average time complexity**.

---

### Input & Output Formats & Constraints

- **Input Operations:** `LRUCache(capacity)`, `get(key)`, `put(key, value)`
- **Output:** `get` returns integer (`value` or `-1`); `put` returns `void`.
- **Constraints:**
  - $1 \le \text{capacity} \le 3000$
  - $0 \le \text{key} \le 10^4$
  - $0 \le \text{value} \le 10^5$
  - At most $2 \times 10^5$ calls will be made to `get` and `put`.

---

### Key Idea & Intuition

- **Why Dual Data Structures?**
  - **Hash Map:** Provides $O(1)$ access to cache nodes by `key`. But hash maps have no intrinsic order of recency.
  - **Doubly Linked List (DLL):** Allows $O(1)$ deletion of any arbitrary node (given its reference) and $O(1)$ insertion at the head.
- **Sentinel (Dummy) Head and Tail:**
  - Using dummy nodes `head` and `tail` eliminates null-pointer edge case checks when adding the first node or removing the last node.
  - **Head:** Most Recently Used (MRU) side.
  - **Tail:** Least Recently Used (LRU) side.

---

### Solution Approach (Step-by-Step)

1. **Helper `_remove(node)`:**
   - Detach node from doubly linked list:
     `node.prev.next = node.next`
     `node.next.prev = node.prev`
2. **Helper `_insert_head(node)`:**
   - Splice node immediately after sentinel `head`:
     `node.next = head.next`, `node.prev = head`
     `head.next.prev = node`, `head.next = node`
3. **`get(key)`:**
   - If `key not in cache`: return `-1`.
   - Node exists: detach it (`_remove(node)`) and re-insert at head (`_insert_head(node)`).
   - Return `node.val`.
4. **`put(key, value)`:**
   - If `key in cache`: update node's value, detach, and re-insert at head.
   - If `key not in cache`:
     - Create `new_node = Node(key, value)`.
     - If `len(cache) == capacity`:
       - Evict `lru_node = tail.prev`.
       - `_remove(lru_node)`.
       - `del cache[lru_node.key]`.
     - Add `new_node` to `cache` and `_insert_head(new_node)`.

---

### Visual Algorithm Walkthrough

```
Capacity = 2

State: [HEAD] <-> [TAIL]
put(1, 1): [HEAD] <-> [1:1] <-> [TAIL]
put(2, 2): [HEAD] <-> [2:2] <-> [1:1] <-> [TAIL]

get(1): Node 1 accessed -> move to head:
           [HEAD] <-> [1:1] <-> [2:2] <-> [TAIL]

put(3, 3): Capacity full! Evict tail.prev (Node 2):
           [HEAD] <-> [3:3] <-> [1:1] <-> [TAIL]
```

---

### Solved Examples with Multiple Inputs

| Operation Sequence | Cache State `[MRU ... LRU]` | Output / Eviction | Notes |
| :--- | :--- | :--- | :--- |
| `LRUCache(2)` | `[]` | `null` | Capacity initialized to 2 |
| `put(1, 1)` | `[1:1]` | `null` | Insert 1 |
| `put(2, 2)` | `[2:2, 1:1]` | `null` | Insert 2 at head |
| `get(1)` | `[1:1, 2:2]` | `1` | 1 accessed $\to$ moves to head; 2 becomes LRU |
| `put(3, 3)` | `[3:3, 1:1]` | `null` | Capacity exceeded $\to$ evicts 2 (LRU) |
| `get(2)` | `[3:3, 1:1]` | `-1` | Key 2 was evicted |
| `put(4, 4)` | `[4:4, 3:3]` | `null` | Capacity exceeded $\to$ evicts 1 (LRU) |
| `get(1)` | `[4:4, 3:3]` | `-1` | Key 1 was evicted |
| `get(3)` | `[3:3, 4:4]` | `3` | Key 3 found and refreshed to head |
| `get(4)` | `[4:4, 3:3]` | `4` | Key 4 found and refreshed to head |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class Node:
    def __init__(self, key: int = 0, val: int = 0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {}  # key -> Node
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _insert_head(self, node: Node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._insert_head(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._remove(node)
            self._insert_head(node)
        else:
            if len(self.cache) >= self.cap:
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]
                
            new_node = Node(key, value)
            self.cache[key] = new_node
            self._insert_head(new_node)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <unordered_map>

class LRUCache {
private:
    struct Node {
        int key, val;
        Node *prev, *next;
        Node(int k = 0, int v = 0) : key(k), val(v), prev(nullptr), next(nullptr) {}
    };

    int capacity;
    std::unordered_map<int, Node*> cache;
    Node *head, *tail;

    void removeNode(Node* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void insertHead(Node* node) {
        node->next = head->next;
        node->prev = head;
        head->next->prev = node;
        head->next = node;
    }

public:
    LRUCache(int cap) : capacity(cap) {
        head = new Node();
        tail = new Node();
        head->next = tail;
        tail->prev = head;
    }

    int get(int key) {
        if (!cache.count(key)) return -1;
        Node* node = cache[key];
        removeNode(node);
        insertHead(node);
        return node->val;
    }

    void put(int key, int value) {
        if (cache.count(key)) {
            Node* node = cache[key];
            node->val = value;
            removeNode(node);
            insertHead(node);
        } else {
            if (cache.size() >= capacity) {
                Node* lru = tail->prev;
                removeNode(lru);
                cache.erase(lru->key);
                delete lru;
            }
            Node* newNode = new Node(key, value);
            cache[key] = newNode;
            insertHead(newNode);
        }
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.HashMap;
import java.util.Map;

class LRUCache {
    static class Node {
        int key, val;
        Node prev, next;
        Node(int k, int v) { this.key = k; this.val = v; }
    }

    private final int capacity;
    private final Map<Integer, Node> cache;
    private final Node head, tail;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.cache = new HashMap<>();
        this.head = new Node(0, 0);
        this.tail = new Node(0, 0);
        head.next = tail;
        tail.prev = head;
    }

    private void removeNode(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void insertHead(Node node) {
        node.next = head.next;
        node.prev = head;
        head.next.prev = node;
        head.next = node;
    }

    public int get(int key) {
        if (!cache.containsKey(key)) return -1;
        Node node = cache.get(key);
        removeNode(node);
        insertHead(node);
        return node.val;
    }

    public void put(int key, int value) {
        if (cache.containsKey(key)) {
            Node node = cache.get(key);
            node.val = value;
            removeNode(node);
            insertHead(node);
        } else {
            if (cache.size() >= capacity) {
                Node lru = tail.prev;
                removeNode(lru);
                cache.remove(lru.key);
            }
            Node newNode = new Node(key, value);
            cache.put(key, newNode);
            insertHead(newNode);
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** Strict $O(1)$ for both `get` and `put` — Hash map lookup and pointer updates take constant time.
- **Space Complexity:** $O(\text{capacity})$ — Stores at most `capacity` node objects in the DLL and hash map.

---

### Takeaway Pattern & Interview Traps

1. **Why Store Both `key` and `value` in the DLL Node?**
   - When evicting the least recently used node from the tail (`lru = tail.prev`), we must delete its entry from the hash map (`cache.remove(lru.key)`). If the node only stored `val`, finding which key to remove from the hash map would require an $\mathcal{O}(N)$ reverse search!
2. **Sentinel Nodes Simplify Invariant Maintenance:**
   - Always initialize `head` and `tail` dummy sentinels (`head.next = tail`, `tail.prev = head`). This completely eliminates null-checking branches when inserting into an empty list or deleting the last element.
3. **Concurrency / Thread Safety (Interview Follow-Up):**
   - In real-world multithreaded systems, wrap accesses with a `ReentrantReadWriteLock` or use concurrent segmentation (similar to Java's `ConcurrentHashMap`) to prevent race conditions on pointer rewiring.

