---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 863: All Nodes Distance K in Binary Tree"
tags:
  - leetcode
  - coding
  - trees
  - bfs
  - dfs
  - graph
  - amazon
  - google
---

# LeetCode 863: All Nodes Distance K in Binary Tree

**Target Companies:** Amazon (Top Tier Tree/Graph Question), Meta, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Tree to Undirected Graph Conversion / Parent Mapping / Multi-Directional BFS

---

### Problem Statement

Given the `root` of a binary tree, the value of a target node `target`, and an integer `k`, return an array of the values of all nodes that have a distance `k` from the target node.

You can return the answer in **any order**.

---

### Input & Output Formats & Constraints

- **Input:** `root: TreeNode`, `target: TreeNode`, `k: int`
- **Output:** `List[int]` — All node values exactly distance `k` from `target`.
- **Constraints:**
  - The number of nodes in the tree is in the range $[1, 500]$.
  - $0 \le \text{Node.val} \le 500$
  - All the values `Node.val` are **unique**.
  - `target` is the node in the tree.
  - $0 \le k \le 1000$

---

### Key Idea & Intuition

- **The Upward Traversal Barrier:**
  - In a standard binary tree, node pointers only point downwards (`left` and `right`).
  - To find nodes at distance $k$, we must explore in **three directions**:
    1. Down into `left` child.
    2. Down into `right` child.
    3. **Upward into the `parent` node!**
- **Tree to Undirected Graph:**
  - **Step 1 (Parent Mapping):** Run a preorder DFS or BFS from `root` to populate a hash table `parent_map` where `parent_map[node] = parent`.
  - **Step 2 (Radial BFS):** Start a BFS from `target` (distance $0$). Use a `visited` set to prevent backtracking (e.g. going back to child from parent).
  - Expand outward level-by-level across `[node.left, node.right, parent_map[node]]`.
  - When the BFS level reaches `k`, the queue contains all nodes at distance $k$!

---

### Solution Approach (Step-by-Step)

1. **Populate Parent Pointers:**
   - Initialize dictionary/map `parents = {}`.
   - Run DFS: for each `node`, record `parents[node.left] = node` and `parents[node.right] = node`.
2. **Breadth-First Search from Target:**
   - Initialize `queue = deque([target])`.
   - Initialize `visited = {target}`.
   - Maintain `curr_dist = 0`.
3. **Level-Order Expansion:**
   - While `queue` and `curr_dist < k`:
     - For each node in the current level:
       - Check `node.left`: if present and not in `visited`, add to queue and `visited`.
       - Check `node.right`: if present and not in `visited`, add to queue and `visited`.
       - Check `parents.get(node)`: if present and not in `visited`, add to queue and `visited`.
     - Increment `curr_dist += 1`.
4. Return `[node.val for node in queue]`.

---

### Visual Algorithm Walkthrough

```
Binary Tree:
         3
       /   \
     [5]    1
     / \   / \
    6   2 0   8
       / \
      7   4

Target = [5], K = 2

Step 1: Build Parent Pointers:
  parents[5] = 3, parents[6] = 5, parents[2] = 5, parents[1] = 3, etc.

Step 2: Radial BFS from [5]:
Level 0: Queue = [5]

Level 1 (Dist 1):
  - Neighbors of 5: left=6, right=2, parent=3
  - Queue = [6, 2, 3]
  - Visited = {5, 6, 2, 3}

Level 2 (Dist 2 == K):
  - From 6: left=null, right=null, parent=5 (visited)
  - From 2: left=7, right=4, parent=5 (visited) -> Add 7, 4
  - From 3: left=5 (visited), right=1, parent=null -> Add 1
  - Queue = [7, 4, 1]

Distance K reached! Result = [7, 4, 1]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Target Inside Subtree
- **Input:** `root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, k = 2`
- **BFS Level Trace:**
  | Distance | Queue Nodes | Neighbors Inspected | Nodes Enqueued for Next Level |
  | :--- | :--- | :--- | :--- |
  | 0 | `[5]` | `left: 6, right: 2, parent: 3` | `[6, 2, 3]` |
  | 1 | `[6, 2, 3]` | from 6: none; from 2: 7, 4; from 3: 1 | `[7, 4, 1]` |
  | 2 ($== k$) | `[7, 4, 1]` | Target distance reached! | Return values: `[7, 4, 1]` |
- **Output:** `[7, 4, 1]`

#### Example 2: Target is Root, $K = 0$
- **Input:** `root = [1], target = 1, k = 0`
- **Step Trace:**
  - Loop condition `curr_dist < k` ($0 < 0$) is False.
  - Queue contains `[1]`.
- **Output:** `[1]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List, Optional
from collections import deque

class TreeNode:
    def __init__(self, x: int):
        self.val = x
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        if k == 0:
            return [target.val]
            
        parents = {}
        
        # 1. Map parent pointers using DFS
        def map_parents(node: Optional[TreeNode], parent: Optional[TreeNode]) -> None:
            if not node:
                return
            parents[node] = parent
            map_parents(node.left, node)
            map_parents(node.right, node)
            
        map_parents(root, None)
        
        # 2. BFS outward from target
        queue = deque([target])
        visited = {target}
        curr_dist = 0
        
        while queue and curr_dist < k:
            level_size = len(queue)
            for _ in range(level_size):
                curr = queue.popleft()
                
                # Check 3 directions: left, right, parent
                for neighbor in (curr.left, curr.right, parents.get(curr)):
                    if neighbor and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        
            curr_dist += 1
            
        return [node.val for node in queue]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    std::vector<int> distanceK(TreeNode* root, TreeNode* target, int k) {
        if (k == 0) return {target->val};

        std::unordered_map<TreeNode*, TreeNode*> parents;
        mapParents(root, nullptr, parents);

        std::queue<TreeNode*> q;
        std::unordered_set<TreeNode*> visited;

        q.push(target);
        visited.insert(target);
        int currDist = 0;

        while (!q.empty() && currDist < k) {
            int levelSize = q.size();
            for (int i = 0; i < levelSize; ++i) {
                TreeNode* curr = q.front();
                q.pop();

                // Explore left, right, parent
                if (curr->left && !visited.count(curr->left)) {
                    visited.insert(curr->left);
                    q.push(curr->left);
                }
                if (curr->right && !visited.count(curr->right)) {
                    visited.insert(curr->right);
                    q.push(curr->right);
                }
                TreeNode* p = parents[curr];
                if (p && !visited.count(p)) {
                    visited.insert(p);
                    q.push(p);
                }
            }
            currDist++;
        }

        std::vector<int> result;
        while (!q.empty()) {
            result.push_back(q.front()->val);
            q.pop();
        }
        return result;
    }

private:
    void mapParents(TreeNode* node, TreeNode* parent, std::unordered_map<TreeNode*, TreeNode*>& parents) {
        if (!node) return;
        parents[node] = parent;
        mapParents(node->left, node, parents);
        mapParents(node->right, node, parents);
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) { val = x; }
}

class Solution {
    public List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
        if (k == 0) {
            return Collections.singletonList(target.val);
        }

        Map<TreeNode, TreeNode> parents = new HashMap<>();
        mapParents(root, null, parents);

        Queue<TreeNode> queue = new ArrayDeque<>();
        Set<TreeNode> visited = new HashSet<>();

        queue.offer(target);
        visited.add(target);
        int currDist = 0;

        while (!queue.isEmpty() && currDist < k) {
            int levelSize = queue.size();
            for (int i = 0; i < levelSize; i++) {
                TreeNode curr = queue.poll();

                if (curr.left != null && !visited.contains(curr.left)) {
                    visited.add(curr.left);
                    queue.offer(curr.left);
                }
                if (curr.right != null && !visited.contains(curr.right)) {
                    visited.add(curr.right);
                    queue.offer(curr.right);
                }
                TreeNode parent = parents.get(curr);
                if (parent != null && !visited.contains(parent)) {
                    visited.add(parent);
                    queue.offer(parent);
                }
            }
            currDist++;
        }

        List<Integer> result = new ArrayList<>();
        while (!queue.isEmpty()) {
            result.add(queue.poll().val);
        }
        return result;
    }

    private void mapParents(TreeNode node, TreeNode parent, Map<TreeNode, TreeNode> parents) {
        if (node == null) return;
        parents.put(node, parent);
        mapParents(node.left, node, parents);
        mapParents(node.right, node, parents);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ — Building the parent mapping with DFS takes $\mathcal{O}(N)$ time. The radial BFS visits each node and edge at most once, also taking $\mathcal{O}(N)$ time. Overall time is strictly linear $\mathcal{O}(N)$.
- **Space Complexity:** $\mathcal{O}(N)$ — The `parents` hash map stores $N$ entries, the `visited` set holds up to $N$ entries, and the BFS queue holds up to $N$ nodes.

---

### Takeaway Pattern & Interview Traps

1. **Cycle Prevention:** Because trees converted to undirected graphs have bidirectional edges, failure to maintain a `visited` set will cause an infinite ping-pong loop between a child node and its parent.
2. **Edge Case $K = 0$:** If $k = 0$, the only node at distance 0 is `target` itself. Returning immediately avoids redundant BFS operations.
3. **DFS Alternative (No Graph Conversion):** You can also solve this purely with recursive DFS by returning the distance of `target` from each ancestor, then searching the opposite subtree with distance $k - d - 1$. However, the Parent Mapping + BFS approach is far more intuitive, modular, and less prone to off-by-one errors during interviews.