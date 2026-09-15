---
date: "2025-12-24"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 1377: Frog Position After T Seconds"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 1377: Frog Position After T Seconds

Below is a **complete, interview-grade explanation** of **LeetCode 1377 – Frog Position After T Seconds**, structured exactly as requested.

---

## 🧩 LeetCode 1377 — *Frog Position After T Seconds*

### 📌 Problem Statement

You are given an **undirected tree** with `n` nodes labeled `1` to `n`.  
The frog starts at **node 1** at time `t = 0`.

Each second:

* The frog jumps to **one unvisited adjacent node**, chosen **uniformly at random**
* If **no unvisited adjacent nodes** exist, the frog **stays in place**

Given:

* `n` nodes
* `edges` defining the tree
* time `t`
* `target` node

Return the **probability** that the frog is on `target` after `t` seconds.

---

## 🔑 Key Observations & Core Concepts

### 1. **Tree ⇒ Unique Path**

* There is **exactly one path** from node `1` to any node
* Probability of reaching `target` depends only on this path

---

### 2. **Uniform Probability Distribution**

* At each node, probability is divided by the number of **unvisited children**
* Parent node is excluded once visited

---

### 3. **Stopping Rule**

* If the frog reaches a node with **no unvisited neighbors**, it **must stay**
* This is critical when `t` is **greater than depth of target**

---

### 4. **DFS with Probability Propagation**

We perform DFS while tracking:

* `current_node`
* `current_time`
* `current_probability`

We stop DFS when:

* `time > t`
* or we reach `target`

---

### 5. **Critical Edge Case**

If the frog reaches `target` **before** time `t`:

* Probability is valid **only if the target has no unvisited neighbors**
* Otherwise, frog must jump away → probability becomes `0`

---

## 🧠 Conceptual Illustration

![Image](https://www.researchgate.net/publication/258350821/figure/fig4/AS%3A341356728864770%401458397194155/Data-on-red-eyed-tree-frogs-size-total-length-in-mm-vs-time-in-days-Points-represent.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2AcpvBj8a_cCDKSZOhKxVSTg.jpeg)

![Image](https://assets.leetcode.com/uploads/2021/12/21/frog2.jpg)

**Example Tree**

```
        1
       / \
      2   3
     /
    4
```

Target = `4`, `t = 2`

* `1 → 2 → 4`
* Each step probability = `1 / available_children`
* Final probability = `1/2 * 1 = 0.5`

---

## 🧪 Worked Example

### Input

```
n = 7
edges = [[1,2],[1,3],[1,7],[2,4],[2,6],[3,5]]
t = 2
target = 4
```

### Step-by-Step Probability Flow

| Time | Node | Available Moves | Probability |
| --- | --- | --- | --- |
| 0 | 1 | 3 | 1 |
| 1 | 2 | 2 | 1/3 |
| 2 | 4 | 0 | **1/6** |

✔ Frog stops at node `4`  
✔ No unvisited neighbors  
✔ **Answer = 1/6**

---

## 🧠 Algorithm (DFS)

### High-Level Steps

1. Build adjacency list
2. DFS from node `1`
3. Track visited nodes
4. Divide probability among children
5. Apply stopping condition at `target`

---

## 🧑‍💻 Python 3 Solution (With Typing)

```python
from typing import List
from collections import defaultdict

class Solution:
    def frogPosition(
        self,
        n: int,
        edges: List[List[int]],
        t: int,
        target: int
    ) -> float:

        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        visited = set()

        def dfs(node: int, time: int, prob: float) -> float:
            visited.add(node)

            # If time exceeds t, invalid path
            if time > t:
                return 0.0

            # If target reached
            if node == target:
                # Valid only if time == t OR frog cannot move further
                if time == t or len(graph[node]) == len(visited):
                    return prob
                return 0.0

            # Count unvisited neighbors
            children = [nei for nei in graph[node] if nei not in visited]
            if not children:
                return 0.0

            next_prob = prob / len(children)
            ans = 0.0

            for child in children:
                ans += dfs(child, time + 1, next_prob)

            return ans

        return dfs(1, 0, 1.0)
```

---

## ⏱ Complexity Analysis

| Metric | Value |
| --- | --- |
| Time | **O(n)** |
| Space | **O(n)** |
| Traversal | DFS |
| Graph | Tree (acyclic) |

---

## 🎯 Final Takeaways

* Tree structure ⇒ **single path probability**
* **Stopping condition** is the most common mistake
* DFS is natural due to time-based depth
* Parent exclusion avoids revisiting

---

If you want:

* BFS version
* Mathematical probability derivation
* Visualization with recursion stack
* Dry-run on a tricky edge case

Tell me and I’ll extend this.