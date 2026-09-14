---
date: "2026-04-06"
type: lld-question
difficulty: easy
status: active
tags: [lld, interview-prep, hashmap, data-structure]
---

# Design a HashMap

## 1. Problem Statement
Design a HashMap from scratch with put, get, delete in O(1) average time. Handle collisions with chaining.

## 2. Key Implementation (Python)

```python
class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None

class MyHashMap:
    def __init__(self, capacity: int = 16, load_factor: float = 0.75):
        self.capacity = capacity
        self.load_factor = load_factor
        self.size = 0
        self.buckets = [None] * capacity

    def _hash(self, key) -> int:
        return hash(key) % self.capacity

    def put(self, key, value):
        idx = self._hash(key)
        node = self.buckets[idx]

        # Check if key exists → update
        while node:
            if node.key == key:
                node.value = value
                return
            node = node.next

        # Insert at head of chain
        new_node = Node(key, value)
        new_node.next = self.buckets[idx]
        self.buckets[idx] = new_node
        self.size += 1

        # Resize if load factor exceeded
        if self.size / self.capacity > self.load_factor:
            self._resize()

    def get(self, key):
        idx = self._hash(key)
        node = self.buckets[idx]
        while node:
            if node.key == key:
                return node.value
            node = node.next
        return None

    def remove(self, key) -> bool:
        idx = self._hash(key)
        node = self.buckets[idx]
        prev = None
        while node:
            if node.key == key:
                if prev:
                    prev.next = node.next
                else:
                    self.buckets[idx] = node.next
                self.size -= 1
                return True
            prev = node
            node = node.next
        return False

    def _resize(self):
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [None] * self.capacity
        self.size = 0
        for node in old_buckets:
            while node:
                self.put(node.key, node.value)
                node = node.next

    def __contains__(self, key):
        return self.get(key) is not None

    def __len__(self):
        return self.size
```

## 3. Java Implementation

```java
public class MyHashMap<K, V> {
    private static class Node<K, V> {
        K key; V value; Node<K, V> next;
        Node(K key, V value) { this.key = key; this.value = value; }
    }

    private Node<K, V>[] buckets;
    private int size;
    private static final float LOAD_FACTOR = 0.75f;

    @SuppressWarnings("unchecked")
    public MyHashMap(int capacity) {
        buckets = new Node[capacity];
    }

    public void put(K key, V value) {
        int idx = Math.abs(key.hashCode()) % buckets.length;
        Node<K, V> node = buckets[idx];
        while (node != null) {
            if (node.key.equals(key)) { node.value = value; return; }
            node = node.next;
        }
        Node<K, V> newNode = new Node<>(key, value);
        newNode.next = buckets[idx];
        buckets[idx] = newNode;
        size++;
        if ((float) size / buckets.length > LOAD_FACTOR) resize();
    }

    public V get(K key) {
        int idx = Math.abs(key.hashCode()) % buckets.length;
        Node<K, V> node = buckets[idx];
        while (node != null) {
            if (node.key.equals(key)) return node.value;
            node = node.next;
        }
        return null;
    }
}
```

## 4. Collision Resolution Strategies
| Strategy | How | Pros | Cons |
|----------|-----|------|------|
| **Chaining** | Linked list per bucket | Simple, handles high load | Cache unfriendly |
| **Open Addressing (Linear Probing)** | Next available slot | Cache friendly | Clustering |
| **Double Hashing** | Second hash for step | Reduces clustering | Complex |
| **Robin Hood** | Steal from rich buckets | Good variance | Complex |

## 5. Follow-ups
- **Thread-safe?** `ConcurrentHashMap` — segment locking.
- **Java TreeMap buckets?** Java 8+ converts chains to balanced trees when >8 nodes.
- **Perfect hashing?** If keys are known in advance, zero collisions.

---

**Related:** [[01 - Strategy Pattern]]
