---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 347: Top K Frequent Elements"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - heap
  - amazon
  - google
---

# LeetCode 347: Top K Frequent Elements

**Target Companies:** Amazon (Top #1 Classic), Google, Meta  
**Difficulty:** Medium  
**Topic:** Bucket Sort in O(N) / Min-Heap

---

### Problem Statement

Given an integer array `nums` and an integer `k`, return the `k` most frequent elements. You may return the answer in **any order**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `k: int`
- **Output:** `List[int]` of length `k`
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $-10^4 \le \text{nums}[i] \le 10^4$
  - $k$ is in the range $[1, \text{unique elements}]$.

---

### Key Idea & Intuition

- **Linear Time Bucket Sort:**
  - Frequency of any element cannot exceed $N = \text{len}(nums)$.
  - Create buckets `buckets = [[] for _ in range(N + 1)]` where index represents frequency.
  - Count frequencies using a hash map `count = Counter(nums)`.
  - Put each number `num` into `buckets[count[num]]`.
  - Iterate from the highest frequency bucket ($N$) down to 1, collecting elements until we have $K$ items.
  - Achieves guaranteed **$O(N)$** runtime!

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import Counter

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        n = len(nums)
        buckets = [[] for _ in range(n + 1)]
        
        for num, freq in count.items():
            buckets[freq].append(num)
            
        res = []
        for freq in range(n, 0, -1):
            for num in buckets[freq]:
                res.append(num)
                if len(res) == k:
                    return res
        return res
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_map>

class Solution {
public:
    std::vector<int> topKFrequent(std::vector<int>& nums, int k) {
        std::unordered_map<int, int> count;
        for (int num : nums) count[num]++;

        int n = nums.size();
        std::vector<std::vector<int>> buckets(n + 1);
        for (auto& [num, freq] : count) {
            buckets[freq].push_back(num);
        }

        std::vector<int> res;
        for (int freq = n; freq >= 1 && res.size() < k; --freq) {
            for (int num : buckets[freq]) {
                res.push_back(num);
                if (res.size() == k) return res;
            }
        }
        return res;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> count = new HashMap<>();
        for (int num : nums) {
            count.put(num, count.getOrDefault(num, 0) + 1);
        }

        int n = nums.length;
        List<Integer>[] buckets = new List[n + 1];
        for (int key : count.keySet()) {
            int freq = count.get(key);
            if (buckets[freq] == null) {
                buckets[freq] = new ArrayList<>();
            }
            buckets[freq].add(key);
        }

        int[] res = new int[k];
        int idx = 0;
        for (int freq = n; freq >= 1 && idx < k; freq--) {
            if (buckets[freq] != null) {
                for (int num : buckets[freq]) {
                    res[idx++] = num;
                    if (idx == k) return res;
                }
            }
        }
        return res;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Linear hash map counting + linear bucket accumulation.
- **Space Complexity:** $O(N)$ for the frequency map and buckets.
