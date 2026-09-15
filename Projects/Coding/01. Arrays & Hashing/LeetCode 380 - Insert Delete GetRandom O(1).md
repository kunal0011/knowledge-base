---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 380: Insert Delete GetRandom O(1)"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - amazon
  - google
---

# LeetCode 380: Insert Delete GetRandom O(1)

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Arrays & Hashing (Design)

---

### Problem Statement

Implement the `RandomizedSet` class:
- `RandomizedSet()`: Initializes the `RandomizedSet` object.
- `bool insert(int val)`: Inserts an item `val` into the set if not present. Returns `true` if the item was not present, `false` otherwise.
- `bool remove(int val)`: Removes an item `val` from the set if present. Returns `true` if the item was present, `false` otherwise.
- `int getRandom()`: Returns a random element from the current set of elements (it's guaranteed that at least one element exists when this method is called). Each element must have the **same probability** of being returned.

You must implement the functions of the class such that each function works in **average $O(1)$ time complexity**.

---

### Input & Output Formats & Constraints

- **Input Operations:** Calls to `insert`, `remove`, `getRandom`
- **Output:** Boolean for `insert`/`remove`, integer for `getRandom`
- **Constraints:**
  - $-2^{31} \le 	ext{val} \le 2^{31} - 1$
  - At most $2 	imes 10^5$ calls will be made to `insert`, `remove`, and `getRandom`.
  - There will be at least one element in the data structure when `getRandom` is called.

---

### Key Idea & Intuition

- **The Tradeoff:**
  - A **Hash Map** provides $O(1)$ insert and $O(1)$ delete, but cannot pick a uniform random key in $O(1)$ time (keys are not indexed contiguously).
  - A **Dynamic Array (Vector)** provides $O(1)$ random access by index via `random.randint(0, len-1)`, but removing an arbitrary element takes $O(N)$ due to shifting.
- **The Core Breakthrough (Swap with Last):**
  - Combine an **Array** (for contiguous storage and $O(1)$ random access) with a **Hash Map** (storing `val -> index in array`).
  - To delete `val` in $O(1)$ without shifting:
    1. Look up the index of `val` in the hash map: `idx = val_to_idx[val]`.
    2. Overwrite the element at `idx` in the array with the **last element** of the array.
    3. Update the hash map for that last element: `val_to_idx[last_val] = idx`.
    4. Pop the last element from the array in $O(1)$.
    5. Delete `val` from the hash map.

---

### Solution Approach (Step-by-Step)

1. **Initialization:**
   - Maintain `nums = []` (dynamic array) and `val_to_idx = {}` (hash map).
2. **`insert(val)`:**
   - If `val` in `val_to_idx`, return `False`.
   - Append `val` to `nums`.
   - Record `val_to_idx[val] = len(nums) - 1`.
   - Return `True`.
3. **`remove(val)`:**
   - If `val` not in `val_to_idx`, return `False`.
   - Get `idx = val_to_idx[val]` and `last_val = nums[-1]`.
   - Overwrite `nums[idx] = last_val`.
   - Update `val_to_idx[last_val] = idx`.
   - Pop `nums.pop()`.
   - Remove `del val_to_idx[val]`.
   - Return `True`.
4. **`getRandom()`:**
   - Return `random.choice(nums)` (or `nums[random.randint(0, len(nums) - 1)]`).

---

### Visual Algorithm Walkthrough

```
Initial State: nums = [10, 20, 30, 40], val_to_idx = {10:0, 20:1, 30:2, 40:3}

Operation: remove(20)
Step 1: Locate 20 at index 1. Last element is 40 at index 3.
Step 2: Copy last element (40) into index 1:
        nums: [10, 40, 30, 40]
Step 3: Update map for 40: val_to_idx[40] = 1
Step 4: Pop array end:
        nums: [10, 40, 30]
Step 5: Delete 20 from map:
        val_to_idx: {10:0, 40:1, 30:2}

Result: O(1) removal with contiguous array intact!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Flow
| Operation | Input | Array State `nums` | Hash Map `val_to_idx` | Return |
|:---|:---|:---|:---|:---:|
| `RandomizedSet()` | - | `[]` | `{}` | `null` |
| `insert(1)` | `1` | `[1]` | `{1: 0}` | `true` |
| `remove(2)` | `2` | `[1]` | `{1: 0}` | `false` |
| `insert(2)` | `2` | `[1, 2]` | `{1: 0, 2: 1}` | `true` |
| `getRandom()` | - | `[1, 2]` | `{1: 0, 2: 1}` | `1` or `2` (50% each) |
| `remove(1)` | `1` | `[2]` | `{2: 0}` | `true` |
| `insert(2)` | `2` | `[2]` | `{2: 0}` | `false` |
| `getRandom()` | - | `[2]` | `{2: 0}` | `2` (100%) |

#### Example 2: Removing the Only Element / Removing the Last Element
- **Input:** `insert(5)`, `remove(5)`
- **Trace:**
  - `nums = [5]`, `val_to_idx = {5: 0}`
  - `idx = 0`, `last_val = 5`.
  - `nums[0] = 5`, `val_to_idx[5] = 0`.
  - `nums.pop()` -> `nums = []`.
  - `del val_to_idx[5]` -> `{}`.
  - Correctly handles single-element boundary without indexing crash.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
import random

class RandomizedSet:
    def __init__(self):
        self.nums = []
        self.val_to_idx = {}

    def insert(self, val: int) -> bool:
        if val in self.val_to_idx:
            return False
        self.val_to_idx[val] = len(self.nums)
        self.nums.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.val_to_idx:
            return False
        idx = self.val_to_idx[val]
        last_val = self.nums[-1]
        
        # Move last element to the slot of the element to delete
        self.nums[idx] = last_val
        self.val_to_idx[last_val] = idx
        
        # Pop from array and remove from map
        self.nums.pop()
        del self.val_to_idx[val]
        return True

    def getRandom(self) -> int:
        return random.choice(self.nums)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_map>
#include <cstdlib>

class RandomizedSet {
private:
    std::vector<int> nums;
    std::unordered_map<int, int> valToIdx;

public:
    RandomizedSet() {}

    bool insert(int val) {
        if (valToIdx.count(val)) return false;
        valToIdx[val] = nums.size();
        nums.push_back(val);
        return true;
    }

    bool remove(int val) {
        if (!valToIdx.count(val)) return false;
        int idx = valToIdx[val];
        int lastVal = nums.back();

        // Swap with last element
        nums[idx] = lastVal;
        valToIdx[lastVal] = idx;

        // Pop last element
        nums.pop_back();
        valToIdx.erase(val);
        return true;
    }

    int getRandom() {
        int randIdx = std::rand() % nums.size();
        return nums[randIdx];
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class RandomizedSet {
    private final List<Integer> nums;
    private final Map<Integer, Integer> valToIdx;
    private final Random rand;

    public RandomizedSet() {
        this.nums = new ArrayList<>();
        this.valToIdx = new HashMap<>();
        this.rand = new Random();
    }

    public boolean insert(int val) {
        if (valToIdx.containsKey(val)) return false;
        valToIdx.put(val, nums.size());
        nums.add(val);
        return true;
    }

    public boolean remove(int val) {
        if (!valToIdx.containsKey(val)) return false;
        int idx = valToIdx.get(val);
        int lastVal = nums.get(nums.size() - 1);

        // Overwrite position with last element
        nums.set(idx, lastVal);
        valToIdx.put(lastVal, idx);

        // Remove the tail
        nums.remove(nums.size() - 1);
        valToIdx.remove(val);
        return true;
    }

    public int getRandom() {
        return nums.get(rand.nextInt(nums.size()));
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - `insert(val)`: Average $O(1)$ for hash map insertion and amortized dynamic array append.
  - `remove(val)`: Average $O(1)$ for hash map lookup, array index overwrite, and array pop.
  - `getRandom()`: Strict $O(1)$ uniform random index lookup.
- **Space Complexity:**
  - $O(N)$ where $N$ is the number of unique elements currently stored in the set.

---

### Takeaway Pattern & Interview Traps

- **Core Pattern:** "Swap-with-Last" deletion technique. Enables $O(1)$ deletion from a dense array when order does not matter.
- **Common Trap:** Updating the hash map *after* popping or deleting: if `val` is itself the last element, setting `val_to_idx[last_val] = idx` before deleting `val` avoids resurrecting the deleted key!
