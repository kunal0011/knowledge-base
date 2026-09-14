---
date: "2026-04-06"
type: lld-question
difficulty: medium
status: active
tags: [lld, interview-prep, linkedin, observer-pattern]
---

# Design LinkedIn

## 1. Problem Statement
Design LLD for a professional networking platform with profiles, connections, messaging, job postings, and feed.

## 2. Key Implementation (Python)

```python
from typing import Dict, List, Set, Optional
from datetime import datetime
from enum import Enum

class ConnectionStatus(Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"

class Profile:
    def __init__(self, user_id: str, name: str, headline: str):
        self.user_id = user_id
        self.name = name
        self.headline = headline
        self.experiences: List[dict] = []
        self.skills: List[str] = []
        self.education: List[dict] = []

class User:
    def __init__(self, user_id: str, name: str, headline: str = ""):
        self.user_id = user_id
        self.profile = Profile(user_id, name, headline)
        self.connections: Set[str] = set()
        self.pending_requests: Dict[str, ConnectionStatus] = {}

    def send_connection_request(self, other: 'User'):
        if other.user_id not in self.connections:
            other.pending_requests[self.user_id] = ConnectionStatus.PENDING
            print(f"📤 {self.profile.name} → {other.profile.name}: Connection request sent")

    def accept_request(self, from_user: 'User'):
        if from_user.user_id in self.pending_requests:
            self.connections.add(from_user.user_id)
            from_user.connections.add(self.user_id)
            self.pending_requests[from_user.user_id] = ConnectionStatus.ACCEPTED
            print(f"✅ {self.profile.name} accepted {from_user.profile.name}")

class JobPosting:
    def __init__(self, job_id: str, company: str, title: str,
                 description: str, skills_required: List[str]):
        self.job_id = job_id
        self.company = company
        self.title = title
        self.description = description
        self.skills_required = skills_required
        self.applicants: List[str] = []

    def apply(self, user: User):
        self.applicants.append(user.user_id)
        print(f"📋 {user.profile.name} applied for {self.title} at {self.company}")

class LinkedInService:
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.jobs: Dict[str, JobPosting] = []

    def register(self, user_id: str, name: str, headline: str = "") -> User:
        user = User(user_id, name, headline)
        self.users[user_id] = user
        return user

    def search_users(self, query: str) -> List[User]:
        q = query.lower()
        return [u for u in self.users.values()
                if q in u.profile.name.lower() or q in u.profile.headline.lower()]

    def search_jobs(self, skill: str) -> List[JobPosting]:
        return [j for j in self.jobs if skill.lower() in [s.lower() for s in j.skills_required]]

    def get_mutual_connections(self, user1_id: str, user2_id: str) -> Set[str]:
        return self.users[user1_id].connections & self.users[user2_id].connections

    def get_recommendations(self, user_id: str) -> List[str]:
        """2nd degree connections — friends of friends"""
        user = self.users[user_id]
        recommendations = set()
        for conn_id in user.connections:
            conn = self.users.get(conn_id)
            if conn:
                recommendations.update(conn.connections)
        recommendations.discard(user_id)
        recommendations -= user.connections
        return list(recommendations)
```

## 3. Patterns: **Observer** (feed, notifications) | **Strategy** (recommendation algorithms) | **State** (connection request lifecycle)

## 4. Follow-ups
- **Endorsements?** Users endorse specific skills of connections.
- **Content feed?** Ranked feed from connections' posts + engagement.
- **People you may know?** Graph-based 2nd/3rd degree connections with mutual count.

---

**Related:** [[02 - Observer Pattern]] | [[01 - Strategy Pattern]] | [[13 - State Pattern]]
