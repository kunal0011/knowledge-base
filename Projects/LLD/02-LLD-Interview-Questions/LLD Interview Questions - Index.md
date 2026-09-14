---
date: "2026-04-06"
type: project
status: active
tags:
  - lld
  - interview-prep
  - object-oriented-design
  - system-design
---

# Top 50 LLD Interview Questions

> [!note] About
> The top 50 low-level design questions asked in software engineering interviews at FAANG/MAANG companies. Each question follows a structured interview-format answer: Requirements → Class Design → Code → Trade-offs → Follow-ups. Implementations provided in Python.

## Interview Approach

```
Step-by-Step LLD Interview Framework:
│
├── 1️⃣  CLARIFY (3 min)
│   └── Ask about scope, scale, core features, actors
│
├── 2️⃣  IDENTIFY (5 min)
│   ├── Core Objects (nouns from requirements)
│   ├── Actions (verbs → methods)
│   └── Relationships (has-a, is-a, uses-a)
│
├── 3️⃣  DESIGN (10 min)
│   ├── Draw class diagram
│   ├── Apply design patterns
│   └── Define interfaces & enums
│
├── 4️⃣  CODE (15-20 min)
│   ├── Implement core classes
│   ├── Handle edge cases
│   └── Add thread safety if needed
│
└── 5️⃣  DISCUSS (5 min)
    ├── SOLID adherence
    ├── Extensibility
    └── Trade-offs & alternatives
```

## All 50 Questions

| # | Question | Difficulty | Key Patterns |
|---|----------|-----------|-------------|
| 1 | [[Design a Parking Lot]] | 🟢 Easy | Strategy, Factory |
| 2 | [[Design an Elevator System]] | 🟡 Medium | State, Strategy, Observer |
| 3 | [[Design a Library Management System]] | 🟢 Easy | Strategy, Observer |
| 4 | [[Design a Vending Machine]] | 🟢 Easy | State, Strategy |
| 5 | [[Design a Tic-Tac-Toe Game]] | 🟢 Easy | Strategy |
| 6 | [[Design a Chess Game]] | 🔴 Hard | Strategy, State, Command |
| 7 | [[Design a Hotel Booking System]] | 🟡 Medium | Strategy, Observer |
| 8 | [[Design a Movie Ticket Booking System]] | 🟡 Medium | Strategy, Observer |
| 9 | [[Design an ATM System]] | 🟡 Medium | State, Command |
| 10 | [[Design a Car Rental System]] | 🟡 Medium | Strategy, Factory, Decorator |
| 11 | [[Design a Social Media Feed]] | 🟡 Medium | Observer, Iterator, Strategy |
| 12 | [[Design a Notification Service (LLD)]] | 🟡 Medium | Observer, Strategy, Factory |
| 13 | [[Design an Online Shopping Cart]] | 🟡 Medium | Strategy, Decorator |
| 14 | [[Design a File System]] | 🟡 Medium | Composite, Iterator |
| 15 | [[Design a Logging Framework]] | 🟡 Medium | Singleton, Strategy, Decorator |
| 16 | [[Design a Cache]] | 🟡 Medium | Strategy, Proxy |
| 17 | [[Design a Pub-Sub Messaging System]] | 🟡 Medium | Observer, Strategy |
| 18 | [[Design a Task Scheduler]] | 🟡 Medium | Command, Strategy |
| 19 | [[Design a Snake and Ladder Game]] | 🟢 Easy | State, Strategy |
| 20 | [[Design Splitwise]] | 🔴 Hard | Strategy, Observer |
| 21 | [[Design Stack Overflow]] | 🟡 Medium | Observer, Strategy |
| 22 | [[Design a Rate Limiter]] | 🟡 Medium | Strategy |
| 23 | [[Design a URL Shortener (LLD)]] | 🟡 Medium | Strategy |
| 24 | [[Design a Payment Processing System]] | 🔴 Hard | Strategy, State, Command |
| 25 | [[Design a Food Delivery System]] | 🟡 Medium | Strategy, State, Observer |
| 26 | [[Design a Ride-Sharing System]] | 🟡 Medium | Strategy, State, Observer |
| 27 | [[Design a Chat Application]] | 🔴 Hard | Observer, Command, Mediator |
| 28 | [[Design Amazon Locker System]] | 🟡 Medium | State, Strategy, Observer |
| 29 | [[Design a HashMap]] | 🟢 Easy | — |
| 30 | [[Design an Inventory Management System]] | 🟡 Medium | Observer, Strategy |
| 31 | [[Design a Spreadsheet]] | 🔴 Hard | Observer, Composite, Command |
| 32 | [[Design an Airline Booking System]] | 🟡 Medium | State, Strategy |
| 33 | [[Design a Music Streaming Service]] | 🟡 Medium | State, Observer, Iterator |
| 34 | [[Design a Text Editor]] | 🟡 Medium | Command, Memento |
| 35 | [[Design a Database Connection Pool]] | 🟡 Medium | Singleton, Proxy |
| 36 | [[Design LinkedIn]] | 🟡 Medium | Observer, Strategy, State |
| 37 | [[Design a Traffic Light System]] | 🟢 Easy | State, Observer |
| 38 | [[Design an Online Auction System]] | 🔴 Hard | Observer, State, Strategy |
| 39 | [[Design a Thread Pool]] | 🟡 Medium | Command, Strategy |
| 40 | [[Design a Deck of Cards]] | 🟢 Easy | Factory, Strategy |
| 41 | [[Design a Meeting Scheduler]] | 🟡 Medium | Observer, Strategy |
| 42 | [[Design a Bowling Alley]] | 🟡 Medium | Strategy |
| 43 | [[Design an Order Management System]] | 🟡 Medium | State, Command, Observer |
| 44 | [[Design an API Gateway]] | 🟡 Medium | Proxy, Decorator, Strategy |
| 45 | [[Design a Todo List Application]] | 🟡 Medium | Command, Composite |
| 46 | [[Design a Stock Exchange System]] | 🔴 Hard | Observer, Strategy, Command |
| 47 | [[Design a Key-Value Store]] | 🟢 Easy | Singleton, Observer |
| 48 | [[Design an Event Bus]] | 🟡 Medium | Observer, Mediator |
| 49 | [[Design a Database Query Builder]] | 🔴 Hard | Builder, Composite, Strategy |
| 50 | [[Design a Circuit Breaker]] | 🔴 Hard | State, Proxy, Decorator |

## Questions by Category

### 🎮 Games & Entertainment
[[Design a Tic-Tac-Toe Game]] | [[Design a Chess Game]] | [[Design a Snake and Ladder Game]] | [[Design a Bowling Alley]] | [[Design a Deck of Cards]]

### 🎫 Booking & Reservation
[[Design a Parking Lot]] | [[Design a Hotel Booking System]] | [[Design a Movie Ticket Booking System]] | [[Design an Airline Booking System]] | [[Design a Meeting Scheduler]]

### 💳 E-Commerce & Payments
[[Design an Online Shopping Cart]] | [[Design a Payment Processing System]] | [[Design Splitwise]] | [[Design an Order Management System]] | [[Design a Todo List Application]]

### 🚗 Transportation
[[Design an Elevator System]] | [[Design a Car Rental System]] | [[Design a Ride-Sharing System]] | [[Design a Traffic Light System]]

### ⚙️ Infrastructure & Systems
[[Design a Cache]] | [[Design a Logging Framework]] | [[Design a Pub-Sub Messaging System]] | [[Design a Task Scheduler]] | [[Design a Rate Limiter]] | [[Design a Database Connection Pool]] | [[Design a HashMap]] | [[Design a File System]] | [[Design a URL Shortener (LLD)]] | [[Design a Thread Pool]] | [[Design an API Gateway]] | [[Design a Key-Value Store]] | [[Design an Event Bus]] | [[Design a Database Query Builder]] | [[Design a Circuit Breaker]]

### 📱 Social & Content
[[Design a Social Media Feed]] | [[Design a Notification Service (LLD)]] | [[Design Stack Overflow]] | [[Design a Music Streaming Service]] | [[Design LinkedIn]] | [[Design a Chat Application]]

### 🍔 Food & Machines
[[Design a Vending Machine]] | [[Design a Food Delivery System]] | [[Design Amazon Locker System]] | [[Design an ATM System]]

### 📈 Finance & Trading
[[Design a Stock Exchange System]] | [[Design an Online Auction System]]

### 📝 Productivity
[[Design a Text Editor]] | [[Design a Spreadsheet]] | [[Design an Inventory Management System]] | [[Design a Library Management System]]

---

> [!tip] Dataview Query
> ```dataview
> TABLE difficulty, tags
> FROM "Projects/LLD/02-LLD-Interview-Questions"
> WHERE type = "lld-question"
> SORT file.name ASC
> ```
