---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 297: Serialize and Deserialize Binary Tree"
tags:
  - leetcode
  - coding
  - trees
  - design
  - dfs
  - amazon
  - google
---

# LeetCode 297: Serialize and Deserialize Binary Tree

**Target Companies:** Meta (Signature Tree System Design Question), Google, Amazon, Microsoft, Uber  
**Difficulty:** Hard  
**Topic:** Tree Serialization & Deserialization / Pre-order Traversal with Null Sentinels / String Tokenization

---

### Problem Statement

Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

---

### Input & Output Formats & Constraints

- **Operations:**
  - `serialize(root: Optional[TreeNode]) -> str`
  - `deserialize(data: str) -> Optional[TreeNode]`
- **Constraints:**
  - The number of nodes in the tree is in the range $[0, 10^4]$.
  - $-1000 \le \text{Node.val} \le 1000$

---

### Key Idea & Intuition

- **Why a Single Traversal Suffices:**
  - Usually, reconstructing a binary tree requires two traversals (e.g. Pre-order + In-order) because an empty subtree cannot be differentiated from a single-child configuration.
  - However, if **null pointers are explicitly preserved** as sentinel markers (e.g., `"#"`, `"N"`, or `"null"`), a **single pre-order traversal** uniquely encodes both the structure and values of any binary tree!
- **Symmetric Recursion:**
  - **Serialization:** Pre-order DFS (`Root -> Left -> Right`).
    - If `node is None`: append `"#"`.
    - Else: append `str(node.val)`, recurse on `node.left`, recurse on `node.right`.
    - Join all tokens with commas `","`.
  - **Deserialization:** Consume tokens in the exact same pre-order sequence using an iterator or queue:
    - Pop the next token.
    - If token is `"#"`, return `None`.
    - Create `root = TreeNode(int(token))`.
    - Recursively deserialize `root.left`.
    - Recursively deserialize `root.right`.
    - Return `root`.

---

### Solution Approach (Step-by-Step)

1. **`serialize(root)`:**
   - Define helper `dfs(node, tokens)`:
     - If `not node`: `tokens.append("#")`; return.
     - `tokens.append(str(node.val))`
     - `dfs(node.left, tokens)`
     - `dfs(node.right, tokens)`
   - Return `",".join(tokens)`.
2. **`deserialize(data)`:**
   - Split `data` by `","` into an iterator or queue `vals`.
   - Define recursive helper `build()`:
     - Pop next value: `val = next(vals)`.
     - If `val == "#"`: return `None`.
     - `node = TreeNode(int(val))`
     - `node.left = build()`
     - `node.right = build()`
     - Return `node`.
   - Return `build()`.

---

### Visual Algorithm Walkthrough

```
Original Tree:
       1
      / \
     2   3
        / \
       4   5

Pre-order Serialization Traversal:
Visit 1: append "1"
  Visit 2: append "2"
    Left of 2 is Null: append "#"
    Right of 2 is Null: append "#"
  Visit 3: append "3"
    Visit 4: append "4"
      Left of 4 is Null: append "#"
      Right of 4 is Null: append "#"
    Visit 5: append "5"
      Left of 5 is Null: append "#"
      Right of 5 is Null: append "#"

Serialized String:
"1,2,#,#,3,4,#,#,5,#,#"

Deserialization Stream:
[1] -> create Node(1)
  [2] -> Node(1).left = Node(2)
    [#] -> Node(2).left = null
    [#] -> Node(2).right = null
  [3] -> Node(1).right = Node(3)
    [4] -> Node(3).left = Node(4)
      [#] -> Node(4).left = null
      [#] -> Node(4).right = null
    [5] -> Node(3).right = Node(5)
      [#] -> Node(5).left = null
      [#] -> Node(5).right = null
Tree fully reconstructed!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Tree
- **Input:** `root = [1,2,3,null,null,4,5]`
- **Serialization Output:** `"1,2,#,#,3,4,#,#,5,#,#"`
- **Deserialization Step Trace:**
  | Token Popped | Node Created | Left Child | Right Child |
  | :--- | :--- | :--- | :--- |
  | `"1"` | `Node(1)` | `build()` $\to$ `Node(2)` | `build()` $\to$ `Node(3)` |
  | `"2"` | `Node(2)` | `"#" -> null` | `"#" -> null` |
  | `"3"` | `Node(3)` | `build()` $\to$ `Node(4)` | `build()` $\to$ `Node(5)` |
  | `"4"` | `Node(4)` | `"#" -> null` | `"#" -> null` |
  | `"5"` | `Node(5)` | `"#" -> null` | `"#" -> null` |
- **Reconstructed Tree:** Identical to original.

#### Example 2: Empty Tree
- **Input:** `root = []`
- **Serialization Output:** `"#"`
- **Deserialization Trace:** Pop `"#"`, returns `None`.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import Optional

class TreeNode:
    def __init__(self, x: int):
        self.val = x
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None

class Codec:
    def serialize(self, root: Optional[TreeNode]) -> str:
        """Encodes a tree to a single string."""
        tokens = []
        
        def dfs(node: Optional[TreeNode]) -> None:
            if not node:
                tokens.append("#")
                return
            tokens.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
            
        dfs(root)
        return ",".join(tokens)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        """Decodes your encoded data to tree."""
        tokens = iter(data.split(","))
        
        def build() -> Optional[TreeNode]:
            val = next(tokens)
            if val == "#":
                return None
            node = TreeNode(int(val))
            node.left = build()
            node.right = build()
            return node
            
        return build()
```

#### 2. C++ (C++17 / STL)
```cpp
#include <string>
#include <sstream>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Codec {
public:
    // Encodes a tree to a single string.
    std::string serialize(TreeNode* root) {
        std::ostringstream out;
        serializeHelper(root, out);
        return out.str();
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(const std::string& data) {
        std::istringstream in(data);
        return deserializeHelper(in);
    }

private:
    void serializeHelper(TreeNode* node, std::ostringstream& out) {
        if (!node) {
            out << "# ";
            return;
        }
        out << node->val << " ";
        serializeHelper(node->left, out);
        serializeHelper(node->right, out);
    }

    TreeNode* deserializeHelper(std::istringstream& in) {
        std::string val;
        if (!(in >> val) || val == "#") {
            return nullptr;
        }

        TreeNode* node = new TreeNode(std::stoi(val));
        node->left = deserializeHelper(in);
        node->right = deserializeHelper(in);
        return node;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) { val = x; }
}

public class Codec {
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        serializeHelper(root, sb);
        return sb.toString();
    }

    private void serializeHelper(TreeNode node, StringBuilder sb) {
        if (node == null) {
            sb.append("#,");
            return;
        }
        sb.append(node.val).append(",");
        serializeHelper(node.left, sb);
        serializeHelper(node.right, sb);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Queue<String> nodes = new LinkedList<>(Arrays.asList(data.split(",")));
        return deserializeHelper(nodes);
    }

    private TreeNode deserializeHelper(Queue<String> nodes) {
        if (nodes.isEmpty()) {
            return null;
        }
        String val = nodes.poll();
        if (val.equals("#")) {
            return null;
        }

        TreeNode node = new TreeNode(Integer.parseInt(val));
        node.left = deserializeHelper(nodes);
        node.right = deserializeHelper(nodes);
        return node;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ — Both serialization and deserialization visit each node and null sentinel exactly once. String tokenization and parsing takes linear time $\mathcal{O}(N)$ proportional to serialized string length.
- **Space Complexity:** $\mathcal{O}(N)$ — The serialized string contains $2N + 1$ tokens (every node has 2 child pointers). The recursion call stack uses $\mathcal{O}(H) \le \mathcal{O}(N)$ memory.

---

### Takeaway Pattern & Interview Traps

1. **Avoid Repeated String Concatenation:** Never use `s += str(node.val) + ","` in a loop or recursion in Python/Java/C++. String concatenation in immutable string languages creates $\mathcal{O}(N^2)$ copies. Always use a list `tokens.append()` with `",".join(tokens)` or `StringBuilder` / `ostringstream`.
2. **Pre-order Slicing vs Iterator:** In Python, passing an `iter(...)` and calling `next(...)` consumes elements in $\mathcal{O}(1)$ without mutating an array or creating index slices.
3. **Alternative Approaches:** Serialization can also be implemented using BFS Level-Order (matching LeetCode's canonical bracket notation `[1,2,3,null,null,4,5]`), but pre-order DFS produces simpler recursive code and cleaner stream processing.