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
---

# LeetCode 355: Design Twitter

Below is a complete, structured treatment of **LeetCode 355 – Design Twitter**, aligned with interview-level expectations.

---

## 1. Problem Statement

Design a simplified version of Twitter that supports the following operations:

1. **postTweet(userId, tweetId)**  
   The user posts a tweet with a unique `tweetId`.
2. **getNewsFeed(userId)**  
   Retrieve the **10 most recent tweet IDs** in the user’s news feed.  
   The news feed consists of tweets posted by:

   * the user themself
   * users they follow  
     Tweets must be ordered from **most recent to least recent**.
3. **follow(followerId, followeeId)**  
   The follower starts following the followee.
4. **unfollow(followerId, followeeId)**  
   The follower stops following the followee.

---

## 2. Key Observations

### Observation 1: Ordering is global, not per user

Tweets must be ordered by **recency across all followed users**, not grouped by user.

➡️ We need a **global timestamp** (monotonically increasing counter).

---

### Observation 2: Each user has their own tweet stream

Each user posts tweets over time, forming an **append-only list**:

```python
user -> [(time, tweetId), (time, tweetId), ...]
```

---

### Observation 3: News Feed = merge K sorted lists

* Each followed user’s tweets are sorted by time.
* We must fetch the **top 10 most recent tweets** across all these lists.

➡️ This is a classic **K-way merge of sorted lists**, where:

* K = number of followed users + self
* We only need the top 10

This strongly suggests a **Priority Queue (Heap)**.

---

## 3. Priority Queue Technique Used

### Why a Heap?

We need to repeatedly extract the **most recent tweet** among multiple users.

### Strategy

1. For each relevant user (self + followees):

   * Take their **most recent tweet**
2. Push these into a **max heap** (simulated using min heap with negative time)
3. Each heap entry contains:

   ```
   (timestamp, tweetId, userId, index_in_user_tweet_list)
   ```
4. When one tweet is popped:

   * Push the **next older tweet** from the same user (if it exists)

### Complexity

* `getNewsFeed`:  
  **O((F + 1) log (F + 1) + 10 log (F + 1))**  
  where F = number of followees
* Efficient because we only extract **10 tweets**

---

## 4. Data Structures Used

```python
self.time        -> global timestamp
self.tweets      -> Dict[userId, List[(time, tweetId)]]
self.following   -> Dict[userId, Set[followeeId]]
```

---

## 5. Python 3 Solution (with typing)

```python
from typing import List, Dict, Set
import heapq
from collections import defaultdict

class Twitter:
    def __init__(self) -> None:
        self.time: int = 0
        self.tweets: Dict[int, List[tuple[int, int]]] = defaultdict(list)
        self.following: Dict[int, Set[int]] = defaultdict(set)

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.time += 1
        self.tweets[userId].append((self.time, tweetId))

    def getNewsFeed(self, userId: int) -> List[int]:
        max_heap: List[tuple[int, int, int, int]] = []

        # include self
        users = self.following[userId] | {userId}

        for u in users:
            if self.tweets[u]:
                time, tweet_id = self.tweets[u][-1]
                index = len(self.tweets[u]) - 1
                # use negative time to simulate max heap
                heapq.heappush(max_heap, (-time, tweet_id, u, index))

        result: List[int] = []

        while max_heap and len(result) < 10:
            neg_time, tweet_id, u, idx = heapq.heappop(max_heap)
            result.append(tweet_id)

            # push next older tweet from same user
            if idx - 1 >= 0:
                time, tweet_id = self.tweets[u][idx - 1]
                heapq.heappush(max_heap, (-time, tweet_id, u, idx - 1))

        return result

    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId:
            self.following[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        self.following[followerId].discard(followeeId)
```

---

## 6. Worked Example (Step-by-Step)

### Operations

```
postTweet(1, 5)
postTweet(1, 3)
postTweet(2, 101)
postTweet(2, 102)
follow(1, 2)
getNewsFeed(1)
```

---

### Internal State

**Global timestamps**

```
1 → (1, 5)
2 → (1, 3)
3 → (2, 101)
4 → (2, 102)
```

**Tweets storage**

```
1: [(1,5), (2,3)]
2: [(3,101), (4,102)]
```

**Following**

```
1 follows {2}
```

---

### Heap Initialization

We add the **latest tweet** of each user:

```
User 1 → (2,3)
User 2 → (4,102)
```

Heap (conceptually):

```
[(4,102), (2,3)]
```

---

### Heap Processing

1️⃣ Pop `(4,102)` → result = `[102]`  
Push next tweet from user 2 → `(3,101)`

Heap:

```
[(3,101), (2,3)]
```

2️⃣ Pop `(3,101)` → result = `[102, 101]`  
No more tweets from user 2

3️⃣ Pop `(2,3)` → result = `[102, 101, 3]`  
Push next from user 1 → `(1,5)`

4️⃣ Pop `(1,5)` → result = `[102, 101, 3, 5]`

---

### Final Output

```
[102, 101, 3, 5]
```

(Ordered from most recent to least recent)

---

## 7. Key Interview Takeaways

* Core insight: **Top-K merge of multiple sorted streams**
* Priority Queue avoids flattening all tweets
* Always include **self-follow**
* Use global timestamp to avoid conflicts
* Heap size bounded by number of followees

If you want, I can also provide:

* Alternate approach comparison (brute force vs heap)
* Time/space optimization discussion
* Follow-up constraints handling (millions of users)

Just tell me.