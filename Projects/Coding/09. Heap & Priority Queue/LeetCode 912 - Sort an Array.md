---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 912: Sort an Array"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
  - sorting
  - amazon
  - google
---

# LeetCode 912: Sort an Array

**Target Companies:** Amazon, Google, Microsoft, Meta, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Heap Sort (In-Place $\mathcal{O}(1)$ Space) / Priority Queue

---

### Problem Statement

Given an array of integers `nums`, sort the array in ascending order and return it.

You must solve the problem **without using any built-in functions** in $\mathcal{O}(N \log N)$ time complexity and with the smallest practical space complexity.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]`, where $1 \le \text{nums.length} \le 5 \times 10^4$.
  - $-5 \times 10^4 \le nums[i] \le 5 \times 10^4$.
- **Output:**
  - `List[int]`: The array sorted in non-decreasing order.
- **Constraints:**
  - No built-in sorting libraries (`sort()`, `Arrays.sort()`, `std::sort()`).
  - Must guarantee $\mathcal{O}(N \log N)$ worst-case time (Quicksort with naive pivot can degrade to $\mathcal{O}(N^2)$).

---

### Key Idea & Intuition

While Merge Sort uses $\mathcal{O}(N)$ auxiliary memory and Quick Sort risks $\mathcal{O}(N^2)$ on adversarial anti-quick-sort test cases, **Heap Sort** provides the ultimate theoretical guarantee:
$$\text{Time: } \mathcal{O}(N \log N) \text{ in all cases (worst, average, best)} \quad \text{and} \quad \text{Auxiliary Space: } \mathcal{O}(1) \text{ strictly in-place!}$$

#### In-Place Heap Sort Algorithm:
A binary heap can be represented compactly as a flat array without pointers:
- For node at index $i$:
  - Left child: $2i + 1$
  - Right child: $2i + 2$
  - Parent: $\lfloor (i - 1) / 2 \rfloor$

The algorithm proceeds in two phases:
1. **Phase 1: Build Max-Heap in $\mathcal{O}(N)$ Time (Bottom-Up Heapify):**
   - Leaf nodes are at indices $\lfloor n/2 \rfloor \dots n-1$ and are trivially valid heaps.
   - Iterate backward from the last non-leaf parent $i = \lfloor n/2 \rfloor - 1$ down to $0$, calling `sift_down(i, n)`:
     - Swap node $i$ with its largest child if that child exceeds $nums[i]$.
     - Continue recursively down the branch.
2. **Phase 2: Extract Maximum and Sort in $\mathcal{O}(N \log N)$ Time:**
   - The global maximum is now at `nums[0]`.
   - Swap `nums[0]` with `nums[n - 1]`. The largest element is now permanently in its sorted position at the end of the array!
   - Sift down the new root `nums[0]` within the reduced active heap size $n - 1$.
   - Repeat for sizes $n-2, n-3, \dots, 1$.
   - The array is sorted strictly in-place with zero extra memory!

---

### Solution Approach (Step-by-Step)

1. **`sift_down(nums, n, i)`:**
   - Identify `largest = i`, `left = 2 * i + 1`, `right = 2 * i + 2`.
   - If `left < n` and `nums[left] > nums[largest]`: `largest = left`.
   - If `right < n` and `nums[right] > nums[largest]`: `largest = right`.
   - If `largest != i`:
     - Swap `nums[i]` and `nums[largest]`.
     - Recursively call `sift_down(nums, n, largest)`.
2. **`sortArray(nums)`:**
   - Let $n = \text{len}(nums)$.
   - Build Max-Heap: loop $i$ from $n // 2 - 1$ down to $0$: `sift_down(nums, n, i)`.
   - Sort array: loop $i$ from $n - 1$ down to $1$:
     - Swap `nums[0]` with `nums[i]`.
     - `sift_down(nums, i, 0)`.
   - Return `nums`.

---

### Visual Algorithm Walkthrough

Sort `nums = [4, 10, 3, 5, 1]`:

```
Phase 1: Build Max-Heap
  n = 5. Non-leaf nodes start at index 5 // 2 - 1 = 1.
  - Sift-down at index 1 (val 10, children 5, 1):
    10 >= 5 and 10 >= 1 -> No change.
  - Sift-down at index 0 (val 4, children 10, 3):
    10 > 4 -> Swap 4 and 10.
    Array becomes: [10, 4, 3, 5, 1]
    Sift-down 4 with children (5, 1): 5 > 4 -> Swap 4 and 5.
    Max-Heap complete: [10, 5, 3, 4, 1]

Phase 2: Repeated Extraction
  - Swap root 10 with last element 1: [1, 5, 3, 4, | 10]
    Sift-down root 1 in size 4:
    Heap becomes: [5, 4, 3, 1, | 10]

  - Swap root 5 with element at index 3 (val 1): [1, 4, 3, | 5, 10]
    Sift-down root 1 in size 3:
    Heap becomes: [4, 1, 3, | 5, 10]

  - Swap root 4 with element at index 2 (val 3): [3, 1, | 4, 5, 10]
    Sift-down root 3 in size 2:
    Heap becomes: [3, 1, | 4, 5, 10]

  - Swap root 3 with element at index 1 (val 1): [1, | 3, 4, 5, 10]

Final Array: [1, 3, 4, 5, 10] (Sorted strictly in-place!)
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Random Permutation

- **Input:** `nums = [5, 2, 3, 1]`
- **Output:** `[1, 2, 3, 5]`

#### Example 2: Array with Duplicates and Negative Numbers

- **Input:** `nums = [5, 1, 1, 2, 0, 0]`
- **Output:** `[0, 0, 1, 1, 2, 5]`

#### Example 3: Already Sorted Array

- **Input:** `nums = [1, 2, 3, 4, 5]`
- **Output:** `[1, 2, 3, 4, 5]`

---

### Multi-Language Implementations

#### Python 3

##### Optimal In-Place Heap Sort ($\mathcal{O}(1)$ Auxiliary Space)
```python
from typing import List

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        n = len(nums)

        def sift_down(heap_size: int, i: int) -> None:
            while True:
                largest = i
                left = 2 * i + 1
                right = 2 * i + 2

                if left < heap_size and nums[left] > nums[largest]:
                    largest = left
                if right < heap_size and nums[right] > nums[largest]:
                    largest = right

                if largest != i:
                    nums[i], nums[largest] = nums[largest], nums[i]
                    i = largest
                else:
                    break

        # Step 1: Build max heap in-place in O(N) time
        for i in range(n // 2 - 1, -1, -1):
            sift_down(n, i)

        # Step 2: Extract elements and sort in-place in O(N log N)
        for i in range(n - 1, 0, -1):
            nums[0], nums[i] = nums[i], nums[0]
            sift_down(i, 0)

        return nums
```

#### C++17

```cpp
#include <vector>
#include <algorithm>

class Solution {
private:
    void siftDown(std::vector<int>& nums, int heap_size, int i) {
        while (true) {
            int largest = i;
            int left = 2 * i + 1;
            int right = 2 * i + 2;

            if (left < heap_size && nums[left] > nums[largest]) {
                largest = left;
            }
            if (right < heap_size && nums[right] > nums[largest]) {
                largest = right;
            }

            if (largest != i) {
                std::swap(nums[i], nums[largest]);
                i = largest;
            } else {
                break;
            }
        }
    }

public:
    std::vector<int> sortArray(std::vector<int>& nums) {
        int n = static_cast<int>(nums.size());

        // Step 1: Build max-heap in O(N)
        for (int i = n / 2 - 1; i >= 0; --i) {
            siftDown(nums, n, i);
        }

        // Step 2: Extract maximum and swap to end
        for (int i = n - 1; i > 0; --i) {
            std::swap(nums[0], nums[i]);
            siftDown(nums, i, 0);
        }

        return nums;
    }
};
```

#### Java

```java
public class Solution {
    private void siftDown(int[] nums, int heapSize, int i) {
        while (true) {
            int largest = i;
            int left = 2 * i + 1;
            int right = 2 * i + 2;

            if (left < heapSize && nums[left] > nums[largest]) {
                largest = left;
            }
            if (right < heapSize && nums[right] > nums[largest]) {
                largest = right;
            }

            if (largest != i) {
                int temp = nums[i];
                nums[i] = nums[largest];
                nums[largest] = temp;
                i = largest;
            } else {
                break;
            }
        }
    }

    public int[] sortArray(int[] nums) {
        int n = nums.length;

        // Step 1: Build max-heap in O(N)
        for (int i = n / 2 - 1; i >= 0; i--) {
            siftDown(nums, n, i);
        }

        // Step 2: In-place sort
        for (int i = n - 1; i > 0; i--) {
            int temp = nums[0];
            nums[0] = nums[i];
            nums[i] = temp;

            siftDown(nums, i, 0);
        }

        return nums;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - **Building the Heap:** $\mathcal{O}(N)$ using bottom-up `sift_down` across non-leaf levels.
  - **Sorting Phase:** $N - 1$ extractions, each requiring $\mathcal{O}(\log N)$ sift-down steps $\implies \mathcal{O}(N \log N)$.
  - **Total Time:** Strictly $\mathcal{O}(N \log N)$ in best, average, and worst cases.
- **Space Complexity:** $\mathcal{O}(1)$
  - Sorting is performed strictly in-place within the existing array without heap or recursion allocations.

---

### Takeaway Pattern & Interview Traps

1. **Iterative Sift-Down vs Recursive:**
   - Writing `sift_down` with a `while True` loop guarantees strictly $\mathcal{O}(1)$ auxiliary space by eliminating call stack overhead.
2. **Why Max-Heap for Ascending Sort?**
   - Swapping the root of a **Max-Heap** to the back of the array places the largest element at index $n - 1$, the second largest at $n - 2$, etc. This naturally arranges the array in **ascending** order without reversing.