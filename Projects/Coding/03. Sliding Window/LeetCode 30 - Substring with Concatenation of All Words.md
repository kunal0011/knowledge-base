---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 30: Substring with Concatenation of All Words"
tags:
  - leetcode
  - coding
  - sliding-window
  - hash-map
  - string
  - amazon
  - google
---

# LeetCode 30: Substring with Concatenation of All Words

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Hard  
**Topic:** Sliding Window / Hash Map / String  

---

### Problem Statement

You are given a string `s` and an array of strings `words`. All strings of `words` are of the **same length**.

A **concatenated string** is a string that exactly contains all the strings of any permutation of `words` concatenated.

- For example, if `words = ["ab","cd","ef"]`, then `"abcdef"`, `"abefcd"`, `"cdabef"`, `"cdefab"`, `"efabcd"`, and `"efcdab"` are all concatenated strings. `"acdbef"` is not a concatenated string because it is not the concatenation of any permutation of `words`.

Return an array of the starting indices of all the concatenated substrings in `s`. You can return the answer in **any order**.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str` / `string` ($1 \le |s| \le 10^4$)
  - `words`: `List[str]` / `vector<string>` / `String[]` ($1 \le |words| \le 5000$, $1 \le |words[i]| \le 30$)
  - All words in `words` have the exact same length: `len(words[i]) == word_len`.
- **Output:**
  - `List[int]` / `vector<int>` / `int[]` — all 0-indexed starting indices in `s` where a concatenated string begins.
- **Constraints:**
  - $1 \le \text{s.length} \le 10^4$
  - $1 \le \text{words.length} \le 5000$
  - $1 \le \text{words}[i]\text{.length} \le 30$
  - `s` and `words[i]` consist of lowercase English letters.

---

### Key Idea & Intuition

Let:
- $L = \text{len}(\text{words}[0])$ be the length of each word.
- $M = \text{len}(\text{words})$ be the total number of words.
- $T = M \times L$ be the total length of the required concatenation.

If we naive-check every index $i \in [0, |s| - T]$ by slicing $M$ words and hashing them, the time is $\mathcal{O}(|s| \times M \times L)$, which could be up to $10^4 \times 5000 \times 30 \approx 1.5 \times 10^9$ operations and will TLE.

#### Optimal Multi-Offset Sliding Window:
Notice that any valid starting index in `s` must belong to one of $L$ equivalence classes modulo $L$: $\{0, 1, 2, \dots, L - 1\}$.
For each fixed offset $i \in [0, L - 1]$:
We can view `s` as a sequence of words of length $L$ starting at offset $i$:
`s[i : i+L], s[i+L : i+2L], s[i+2L : i+3L], ...`

Now the problem becomes equivalent to **Finding All Anagrams of a Fixed Frequency Multi-set (LeetCode 438)**, but operating on word tokens of length $L$ instead of single characters!

For each offset $i \in [0, L - 1]$:
1. Maintain a window $[left, right]$ advancing in steps of $L$.
2. Extract word $w = s[right : right + L]$.
3. If $w$ is not in `words`:
   - Reset the window: all accumulated words are invalid because $w$ breaks the contiguous sequence. Set $left = right + L$, clear the frequency map, and reset matched count to 0.
4. If $w$ is in `words`:
   - Increment count of $w$ in current window `seen[w] += 1`.
   - If `seen[w] > word_freq[w]`, slide $left$ forward by $L$ until the excess occurrence of $w$ is removed, decrementing counts of passed words along the way.
   - If the number of words in current window reaches $M$, record $left$ into the result list! Then advance $left$ by $L$ and decrement the leaving word.

Because each offset processes each word-token at most twice (once entering via $right$, once leaving via $left$), running $L$ parallel offsets gives total time:
$$L \times \mathcal{O}\left(\frac{|s|}{L}\right) \times L = \mathcal{O}(|s| \times L)$$
For $|s| \le 10^4$ and $L \le 30$, $|s| \times L \approx 3 \times 10^5$ operations, which executes in a few milliseconds!

---

### Solution Approach (Step-by-Step)

1. Precompute word frequency map `word_count` from `words`.
2. Let $L = \text{len}(words[0])$ and $M = \text{len}(words)$. If $|s| < M \times L$, return empty list `[]`.
3. Loop through offsets `offset` from $0$ to $L - 1$:
   - Set `left = offset`, `right = offset`, `count = 0`, and `seen = {}`.
   - While `right + L <= len(s)`:
     - `w = s[right : right + L]`
     - `right += L`
     - If `w` is in `word_count`:
       - `seen[w] += 1`
       - `count += 1`
       - While `seen[w] > word_count[w]`:
         - `removed = s[left : left + L]`
         - `seen[removed] -= 1`
         - `count -= 1`
         - `left += L`
       - If `count == M`:
         - Append `left` to `result`.
     - Else (`w` not in `word_count`):
       - `seen.clear()`
       - `count = 0`
       - `left = right`
4. Return `result`.

---

### Visual Algorithm Walkthrough

For `s = "barfoothefoobarman"`, `words = ["foo", "bar"]` ($L = 3, M = 2, M \times L = 6$):
`word_count = {"foo": 1, "bar": 1}`

```
Offsets to check: 0, 1, 2

--- OFFSET 0 ---
left = 0, right = 0
Step 1: right = 0 -> word = "bar" (in map)
  seen = {"bar": 1}, count = 1, right = 3
Step 2: right = 3 -> word = "foo" (in map)
  seen = {"bar": 1, "foo": 1}, count = 2 == M!
  -> FOUND VALID MATCH AT left = 0. Add 0 to result.
  right = 6
Step 3: right = 6 -> word = "the" (not in map)
  Reset: seen = {}, count = 0, left = 9, right = 9
Step 4: right = 9 -> word = "foo" (in map)
  seen = {"foo": 1}, count = 1, right = 12
Step 5: right = 12 -> word = "bar" (in map)
  seen = {"foo": 1, "bar": 1}, count = 2 == M!
  -> FOUND VALID MATCH AT left = 9. Add 9 to result.
  right = 15
Step 6: right = 15 -> word = "man" (not in map)
  Reset: seen = {}, count = 0, left = 18, right = 18. End of s.

--- OFFSET 1 ---
Tokens: "arf", "oot", "hef", "oob", "arm"... None match words dictionary.
No matches found.

--- OFFSET 2 ---
Tokens: "rfo", "oth", "efo", "oba", "rma"... None match words dictionary.
No matches found.

Final Result: [0, 9]
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `s = "barfoothefoobarman"`, `words = ["foo", "bar"]`
- **Output:** `[0, 9]`

#### Example 2 (Duplicate Words in Dictionary):
- **Input:** `s = "wordgoodgoodgoodbestword"`, `words = ["word", "good", "best", "word"]`
- **Expected:** `[]` (since "good" occurs 3 times sequentially, but dictionary only has 1 "good" and 2 "word"s).
- **Output:** `[]`

#### Example 3 (Overlapping Matches):
- **Input:** `s = "barfoofoobarthefoobarman"`, `words = ["bar", "foo", "the"]`
- **Output:** `[6, 9, 12]`

---

### Multi-Language Implementations

#### Python 3
```python
from collections import Counter, defaultdict
from typing import List

class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        if not s or not words:
            return []
        
        word_len = len(words[0])
        num_words = len(words)
        total_len = word_len * num_words
        n = len(s)
        
        if n < total_len:
            return []
            
        word_count = Counter(words)
        result: List[int] = []
        
        # Check all possible offsets: 0, 1, ..., word_len - 1
        for offset in range(word_len):
            left = offset
            right = offset
            seen = defaultdict(int)
            matched_words = 0
            
            while right + word_len <= n:
                w = s[right : right + word_len]
                right += word_len
                
                if w in word_count:
                    seen[w] += 1
                    matched_words += 1
                    
                    # If current word frequency exceeds expectation, shrink from left
                    while seen[w] > word_count[w]:
                        left_w = s[left : left + word_len]
                        seen[left_w] -= 1
                        matched_words -= 1
                        left += word_len
                        
                    # Valid concatenated window found
                    if matched_words == num_words:
                        result.append(left)
                else:
                    # Invalid word: flush window and reset
                    seen.clear()
                    matched_words = 0
                    left = right
                    
        return result
```

#### C++17
```cpp
#include <vector>
#include <string>
#include <unordered_map>
#include <string_view>

class Solution {
public:
    std::vector<int> findSubstring(const std::string& s, const std::vector<std::string>& words) {
        if (s.empty() || words.empty()) return {};
        
        int n = static_cast<int>(s.size());
        int word_len = static_cast<int>(words[0].size());
        int num_words = static_cast<int>(words.size());
        int total_len = word_len * num_words;
        
        if (n < total_len) return {};
        
        std::unordered_map<std::string, int> word_count;
        for (const auto& w : words) {
            word_count[w]++;
        }
        
        std::vector<int> result;
        
        // Check every offset from 0 to word_len - 1
        for (int offset = 0; offset < word_len; ++offset) {
            int left = offset;
            int right = offset;
            int matched = 0;
            std::unordered_map<std::string, int> seen;
            
            while (right + word_len <= n) {
                std::string w = s.substr(right, word_len);
                right += word_len;
                
                auto it = word_count.find(w);
                if (it != word_count.end()) {
                    seen[w]++;
                    matched++;
                    
                    while (seen[w] > it->second) {
                        std::string left_w = s.substr(left, word_len);
                        seen[left_w]--;
                        matched--;
                        left += word_len;
                    }
                    
                    if (matched == num_words) {
                        result.push_back(left);
                    }
                } else {
                    seen.clear();
                    matched = 0;
                    left = right;
                }
            }
        }
        
        return result;
    }
};
```

#### Java 17
```java
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class Solution {
    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> result = new ArrayList<>();
        if (s == null || words == null || words.length == 0 || s.length() == 0) {
            return result;
        }
        
        int n = s.length();
        int wordLen = words[0].length();
        int numWords = words.length;
        int totalLen = wordLen * numWords;
        
        if (n < totalLen) {
            return result;
        }
        
        Map<String, Integer> wordCount = new HashMap<>();
        for (String w : words) {
            wordCount.put(w, wordCount.getOrDefault(w, 0) + 1);
        }
        
        // Iterate through all remainder offsets
        for (int offset = 0; offset < wordLen; offset++) {
            int left = offset;
            int right = offset;
            int matched = 0;
            Map<String, Integer> seen = new HashMap<>();
            
            while (right + wordLen <= n) {
                String sub = s.substring(right, right + wordLen);
                right += wordLen;
                
                if (wordCount.containsKey(sub)) {
                    seen.put(sub, seen.getOrDefault(sub, 0) + 1);
                    matched++;
                    
                    // If word count exceeds requirement, slide left forward
                    while (seen.get(sub) > wordCount.get(sub)) {
                        String leftSub = s.substring(left, left + wordLen);
                        seen.put(leftSub, seen.get(leftSub) - 1);
                        matched--;
                        left += wordLen;
                    }
                    
                    if (matched == numWords) {
                        result.add(left);
                    }
                } else {
                    seen.clear();
                    matched = 0;
                    left = right;
                }
            }
        }
        
        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(|s| \times L)$
  - There are $L$ offsets.
  - In each offset, `right` moves in jumps of $L$ through $\approx \frac{|s|}{L}$ positions.
  - Substring creation takes $\mathcal{O}(L)$.
  - `left` and `right` both visit each word token at most once per offset.
  - Total time: $L \times \frac{|s|}{L} \times \mathcal{O}(L) = \mathcal{O}(|s| \times L)$. With $|s| \le 10^4$ and $L \le 30$, this is $\le 3 \times 10^5$ operations.
- **Space Complexity:** $\mathcal{O}(M \times L)$
  - Storing the hash map `word_count` and `seen` requires space proportional to the number of unique words times word length.

---

### Takeaway Pattern & Interview Traps

- **Modulo Offsets Decomposition:** Whenever a sliding window problem operates on fixed-length tokens of size $L > 1$, always decompose the problem into $L$ independent 1-D sliding windows indexed by offset $\in [0, L - 1]$.
- **Excess Word Eviction:** When `seen[w] > word_count[w]`, you don't need to discard the entire window. Only advance `left` until the previous instance of `w` is evicted.
- **Invalid Word Flush:** When encountering a word not present anywhere in `words`, discard the entire current window immediately (`seen.clear()`, `left = right`), because no contiguous valid permutation can span across an alien word.