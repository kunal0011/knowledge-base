---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 355: Design Twitter"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
  - hash-map
  - design
  - amazon
  - google
---

# LeetCode 355: Design Twitter

**Target Companies:** Amazon, Google, Meta, Twitter/X, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Priority Queue (K-Way Merge) / Hash Map / System Design

---

### Problem Statement

Design a simplified version of Twitter where users can post tweets, follow/unfollow another user, and is able to see the $10$ most recent tweets in the user's news feed.

Implement the `Twitter` class:
- `Twitter()`: Initializes your twitter object.
- `void postTweet(int userId, int tweetId)`: Composes a new tweet with ID `tweetId` by the user `userId`. Each call to this function will be made with a unique `tweetId`.
- `List<Integer> getNewsFeed(int userId)`: Retrieves the $10$ most recent tweet IDs in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user themself. Tweets must be **ordered from most recent to least recent**.
- `void follow(int followerId, int followeeId)`: The user with ID `followerId` started following the user with ID `followeeId`.
- `void unfollow(int followerId, int followeeId)`: The user with ID `followerId` started unfollowing the user with ID `followeeId`.

---

### Input & Output Formats & Constraints

- **Input:**
  - Standard method invocations on the `Twitter` object.
- **Output:**
  - `List[int]` for `getNewsFeed(userId)`, `void` for others.
- **Constraints:**
  - $1 \le userId, followerId, followeeId \le 500$.
  - $0 \le tweetId \le 10^4$.
  - All the tweets have unique IDs.
  - At most $3 \times 10^4$ calls will be made to `postTweet`, `getNewsFeed`, `follow`, and `unfollow`.
  - A user cannot follow themselves.

---

### Key Idea & Intuition

The challenge in designing a scalable feed generator is reconciling **recency across independent streams of events**:
1. Every user posts tweets sequentially over time. Thus, each user's tweet history is an **already sorted list** in chronological order.
2. Generating a user's news feed requires retrieving the $10$ most recent tweets across the user themselves and all their followees ($K$ users total).
3. This is precisely the **$K$-Way Merge** problem (identical to LeetCode 23: Merge $k$ Sorted Lists), truncated at $10$ items:
   - For each followed user (plus self), inspect their most recent tweet.
   - Insert their latest tweet into a **Max-Heap** keyed on a global monotonically increasing timestamp.
   - Extract the most recent tweet from the heap, append to the feed, and insert the next older tweet from that same user into the heap.
   - Terminate once $10$ tweets are collected or the heap empties.

---

### Solution Approach (Step-by-Step)

1. **State Tracking:**
   - `time`: global monotonically increasing integer counter.
   - `tweets`: map from `userId` to list of pairs `(time, tweetId)`.
   - `following`: map from `userId` to set of `followeeId`s.
2. **`postTweet(userId, tweetId)`:**
   - Increment `time += 1`.
   - Append `(time, tweetId)` to `tweets[userId]`.
3. **`getNewsFeed(userId)`:**
   - Union the user's following list with `{userId}` to include self.
   - Max-heap initialization: for each user $u$, push their newest tweet: `(-time, tweetId, u, index)` where `index = len(tweets[u]) - 1`.
   - Loop up to 10 times while heap is non-empty:
     - Pop top tweet `(neg_time, tweetId, u, idx)`.
     - Append `tweetId` to result.
     - If `idx - 1 >= 0`: push next older tweet: `(-tweets[u][idx-1][0], tweets[u][idx-1][1], u, idx - 1)`.
   - Return `result`.
4. **`follow(followerId, followeeId)`:**
   - If `followerId != followeeId`: add to set.
5. **`unfollow(followerId, followeeId)`:**
   - Discard `followeeId` from `following[followerId]`.

---

### Visual Algorithm Walkthrough

```
State:
User 1 follows User 2
User 1 tweets: [t1: 5], [t2: 3]
User 2 tweets: [t3: 101], [t4: 102]

getNewsFeed(1):
Step 1: Gather latest tweet of each source
  User 1 latest: t2 (3)
  User 2 latest: t4 (102)
  Heap: [ (t4, 102, User 2), (t2, 3, User 1) ]

Step 2: Extract top 10
  - Pop (t4, 102). Result = [102]. Push User 2's previous: (t3, 101).
    Heap: [ (t3, 101, User 2), (t2, 3, User 1) ]
  - Pop (t3, 101). Result = [102, 101]. User 2 has no older tweets.
    Heap: [ (t2, 3, User 1) ]
  - Pop (t2, 3). Result = [102, 101, 3]. Push User 1's previous: (t1, 5).
    Heap: [ (t1, 5, User 1) ]
  - Pop (t1, 5). Result = [102, 101, 3, 5]. User 1 has no older tweets.
    Heap empty.

Final News Feed: [102, 101, 3, 5] (Most recent to least recent)
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Follow & Feed

- **Sequence:**
  - `postTweet(1, 5)` $\implies$ User 1: `[5]`
  - `getNewsFeed(1)` $\implies$ `[5]`
  - `follow(1, 2)` $\implies$ 1 follows 2
  - `postTweet(2, 6)` $\implies$ User 2: `[6]`
  - `getNewsFeed(1)` $\implies$ `[6, 5]`
  - `unfollow(1, 2)` $\implies$ 1 unfollows 2
  - `getNewsFeed(1)` $\implies$ `[5]`

#### Example 2: More Than 10 Tweets Available

- User 1 posts 15 tweets: $1 \dots 15$.
- `getNewsFeed(1)` returns the 10 most recent: `[15, 14, 13, 12, 11, 10, 9, 8, 7, 6]`.

---

### Multi-Language Implementations

#### Python 3

```python
import heapq
from collections import defaultdict
from typing import Dict, List, Set, Tuple

class Twitter:
    def __init__(self):
        self.time = 0
        # userId -> list of (timestamp, tweetId)
        self.tweets: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        # followerId -> set of followeeIds
        self.following: Dict[int, Set[int]] = defaultdict(set)

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.time += 1
        self.tweets[userId].append((self.time, tweetId))

    def getNewsFeed(self, userId: int) -> List[int]:
        max_heap = []
        # Include self in sources
        sources = self.following[userId] | {userId}

        # Initialize heap with the latest tweet from each followee
        for u in sources:
            if self.tweets[u]:
                idx = len(self.tweets[u]) - 1
                t, tid = self.tweets[u][idx]
                # Storing (-time, tweetId, user, index_in_user_list)
                heapq.heappush(max_heap, (-t, tid, u, idx))

        result = []
        while max_heap and len(result) < 10:
            neg_t, tid, u, idx = heapq.heappop(max_heap)
            result.append(tid)

            if idx > 0:
                prev_t, prev_tid = self.tweets[u][idx - 1]
                heapq.heappush(max_heap, (-prev_t, prev_tid, u, idx - 1))

        return result

    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId:
            self.following[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        self.following[followerId].discard(followeeId)
```

#### C++17

```cpp
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>

class Twitter {
private:
    struct Tweet {
        int time;
        int id;
    };

    struct HeapNode {
        int time;
        int tweetId;
        int userId;
        int index;

        bool operator<(const HeapNode& other) const {
            return time < other.time; // Max-heap ordered by time
        }
    };

    int timer;
    std::unordered_map<int, std::vector<Tweet>> tweets;
    std::unordered_map<int, std::unordered_set<int>> following;

public:
    Twitter() : timer(0) {}

    void postTweet(int userId, int tweetId) {
        tweets[userId].push_back({++timer, tweetId});
    }

    std::vector<int> getNewsFeed(int userId) {
        std::priority_queue<HeapNode> max_heap;

        // Collect all target users (self + followees)
        std::unordered_set<int> sources = following[userId];
        sources.insert(userId);

        for (int u : sources) {
            auto it = tweets.find(u);
            if (it != tweets.end() && !it->second.empty()) {
                int last_idx = static_cast<int>(it->second.size()) - 1;
                max_heap.push({it->second[last_idx].time, it->second[last_idx].id, u, last_idx});
            }
        }

        std::vector<int> feed;
        while (!max_heap.empty() && feed.size() < 10) {
            HeapNode top = max_heap.top();
            max_heap.pop();
            feed.push_back(top.tweetId);

            if (top.index > 0) {
                int next_idx = top.index - 1;
                max_heap.push({tweets[top.userId][next_idx].time, tweets[top.userId][next_idx].id, top.userId, next_idx});
            }
        }

        return feed;
    }

    void follow(int followerId, int followeeId) {
        if (followerId != followeeId) {
            following[followerId].insert(followeeId);
        }
    }

    void unfollow(int followerId, int followeeId) {
        following[followerId].erase(followeeId);
    }
};
```

#### Java

```java
import java.util.*;

public class Twitter {
    private static class Tweet {
        int time;
        int id;
        Tweet(int time, int id) {
            this.time = time;
            this.id = id;
        }
    }

    private static class FeedNode {
        int time;
        int tweetId;
        int userId;
        int index;

        FeedNode(int time, int tweetId, int userId, int index) {
            this.time = time;
            this.tweetId = tweetId;
            this.userId = userId;
            this.index = index;
        }
    }

    private int timer;
    private Map<Integer, List<Tweet>> tweets;
    private Map<Integer, Set<Integer>> following;

    public Twitter() {
        this.timer = 0;
        this.tweets = new HashMap<>();
        this.following = new HashMap<>();
    }

    public void postTweet(int userId, int tweetId) {
        tweets.computeIfAbsent(userId, k -> new ArrayList<>()).add(new Tweet(++timer, tweetId));
    }

    public List<Integer> getNewsFeed(int userId) {
        PriorityQueue<FeedNode> maxHeap = new PriorityQueue<>((a, b) -> Integer.compare(b.time, a.time));

        Set<Integer> sources = new HashSet<>(following.getOrDefault(userId, Collections.emptySet()));
        sources.add(userId);

        for (int u : sources) {
            List<Tweet> userTweets = tweets.get(u);
            if (userTweets != null && !userTweets.isEmpty()) {
                int lastIdx = userTweets.size() - 1;
                Tweet t = userTweets.get(lastIdx);
                maxHeap.offer(new FeedNode(t.time, t.id, u, lastIdx));
            }
        }

        List<Integer> feed = new ArrayList<>();
        while (!maxHeap.isEmpty() && feed.size() < 10) {
            FeedNode node = maxHeap.poll();
            feed.add(node.tweetId);

            if (node.index > 0) {
                int nextIdx = node.index - 1;
                Tweet prevTweet = tweets.get(node.userId).get(nextIdx);
                maxHeap.offer(new FeedNode(prevTweet.time, prevTweet.id, node.userId, nextIdx));
            }
        }

        return feed;
    }

    public void follow(int followerId, int followeeId) {
        if (followerId != followeeId) {
            following.computeIfAbsent(followerId, k -> new HashSet<>()).add(followeeId);
        }
    }

    public void unfollow(int followerId, int followeeId) {
        Set<Integer> set = following.get(followerId);
        if (set != null) {
            set.remove(followeeId);
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - `postTweet`: $\mathcal{O}(1)$ (append to dynamic array).
  - `follow` / `unfollow`: $\mathcal{O}(1)$ (hash set insertion/deletion).
  - `getNewsFeed`: $\mathcal{O}(F \log F + 10 \log F)$ where $F$ is the number of users followed by the user.
    - Initializing the heap with $F + 1$ elements takes $\mathcal{O}(F \log F)$.
    - Extracting at most 10 tweets takes $10 \times \mathcal{O}(\log F)$.
- **Space Complexity:** $\mathcal{O}(U + T)$
  - Storing user follows and total posted tweets $T$.

---

### Takeaway Pattern & Interview Traps

1. **Self-Follow Requirement:**
   - A user's news feed must include their own tweets. Always include `userId` in `sources`.
2. **K-Way Merge on Demand (Fan-out on Read):**
   - In distributed systems, this is known as the **Pull model / Fan-out on Read**. It avoids duplicating tweet IDs into every follower's timeline when posting, making `postTweet` strictly $\mathcal{O}(1)$.