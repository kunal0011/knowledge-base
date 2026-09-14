# Low-Level Design (LLD) & Design Patterns Masterclass

[![Design Patterns](https://img.shields.io/badge/Design%20Patterns-14%20Covered-blue.svg)](#part-1-head-first-design-patterns)
[![LLD Questions](https://img.shields.io/badge/LLD%20Questions-50%20Systems-success.svg)](#part-2-top-50-lld-interview-questions)
[![Languages](https://img.shields.io/badge/Languages-Python%20%7C%20Java-orange.svg)](#overview)
[![Level](https://img.shields.io/badge/Target%20Level-SDE--2%20%7C%20Staff%20%28L5%2FL6%29-purple.svg)](#interview-framework)
[![Clean Code](https://img.shields.io/badge/Standard-SOLID%20%26%20Clean%20Architecture-brightgreen.svg)](#solid-principles-quick-reference)

> A production-grade, interview-tested curriculum for **Low-Level Design (LLD)**, **Object-Oriented Design (OOD)**, and **Software Architecture Patterns**. Features all **14 Head First Design Patterns** and the **Top 50 FAANG/MAANG LLD Interview Questions**, with class diagrams, complete implementations in Python and Java, edge-case analysis, and concurrency handling.

---

## 🧭 Repository Structure

```
LLD/
├── 01-Design-Patterns/                       # 14 Head First & GoF Design Patterns
│   ├── Design Patterns - Index.md            # Patterns navigation hub & taxonomy
│   ├── 01-Strategy-Pattern/                  # Favor composition over inheritance
│   ├── 02-Observer-Pattern/                  # Loose coupling & event dispatch
│   ├── 03-Decorator-Pattern/                 # Open-Closed runtime behavior extension
│   ├── 04-Factory-Method-Pattern/            # Subclass-driven object instantiation
│   ├── 05-Abstract-Factory-Pattern/          # Families of related/dependent objects
│   ├── 06-Singleton-Pattern/                 # Thread-safe double-checked locking
│   ├── 07-Command-Pattern/                   # Request encapsulation, undo/redo
│   ├── 08-Adapter-Pattern/                   # Interface translation & bridging
│   ├── 09-Facade-Pattern/                    # Unified high-level subsystem interface
│   ├── 10-Template-Method-Pattern/           # Algorithm skeleton & Hollywood principle
│   ├── 11-Iterator-Pattern/                  # Sequential traversal abstraction
│   ├── 12-Composite-Pattern/                 # Part-whole tree hierarchies
│   ├── 13-State-Pattern/                     # State machine encapsulation
│   └── 14-Proxy-Pattern/                     # Surrogate & access control
│
└── 02-LLD-Interview-Questions/               # Top 50 FAANG/MAANG LLD Systems
    ├── LLD Interview Questions - Index.md    # Master curriculum index & domain categorization
    ├── 01-Parking-Lot/                       # Multi-floor spot allocation & pricing strategies
    ├── 02-Elevator-System/                   # SCAN/LOOK dispatching & state machines
    ├── 03-Library-Management/                # Catalog search, lending, fine calculations
    ├── 04-Vending-Machine/                   # State pattern coin/item transaction handling
    ├── ...                                   # 50 complete interview systems
    └── 50-Circuit-Breaker/                   # Resilient failure trip/half-open state pattern
```

---

## 📚 Part 1: Head First Design Patterns (14 Patterns)

Each design pattern includes formal problem definitions, class diagrams (Mermaid), SOLID evaluations, and full implementations in both **Python** and **Java**:

| # | Pattern | Category | Core Architectural Principle | Documentation & Code |
|:---:|:---|:---:|:---|:---|
| **01** | **Strategy** | Behavioral | Favor composition over inheritance; encapsulate interchangeable algorithms | [01 - Strategy Pattern.md](01-Design-Patterns/01-Strategy-Pattern/01%20-%20Strategy%20Pattern.md) |
| **02** | **Observer** | Behavioral | Strive for loosely coupled designs between interacting objects | [02 - Observer Pattern.md](01-Design-Patterns/02-Observer-Pattern/02%20-%20Observer%20Pattern.md) |
| **03** | **Decorator** | Structural | Open-Closed Principle (OCP): open for extension, closed for modification | [03 - Decorator Pattern.md](01-Design-Patterns/03-Decorator-Pattern/03%20-%20Decorator%20Pattern.md) |
| **04** | **Factory Method** | Creational | Dependency Inversion: depend upon abstractions, not concrete classes | [04 - Factory Method Pattern.md](01-Design-Patterns/04-Factory-Method-Pattern/04%20-%20Factory%20Method%20Pattern.md) |
| **05** | **Abstract Factory** | Creational | Encapsulate creation of product families without binding to concrete classes | [05 - Abstract Factory Pattern.md](01-Design-Patterns/05-Abstract-Factory-Pattern/05%20-%20Abstract%20Factory%20Pattern.md) |
| **06** | **Singleton** | Creational | Ensure a class has only one instance with thread-safe double-checked locking | [06 - Singleton Pattern.md](01-Design-Patterns/06-Singleton-Pattern/06%20-%20Singleton%20Pattern.md) |
| **07** | **Command** | Behavioral | Encapsulate request as an object, allowing queuing, logging, and undo/redo | [07 - Command Pattern.md](01-Design-Patterns/07-Command-Pattern/07%20-%20Command%20Pattern.md) |
| **08** | **Adapter** | Structural | Convert the interface of a class into another interface clients expect | [08 - Adapter Pattern.md](01-Design-Patterns/08-Adapter-Pattern/08%20-%20Adapter%20Pattern.md) |
| **09** | **Facade** | Structural | Provide a unified, simplified interface to a complex set of subsystem interfaces | [09 - Facade Pattern.md](01-Design-Patterns/09-Facade-Pattern/09%20-%20Facade%20Pattern.md) |
| **10** | **Template Method** | Behavioral | Define skeleton of an algorithm; let subclasses redefine steps without changing structure | [10 - Template Method Pattern.md](01-Design-Patterns/10-Template-Method-Pattern/10%20-%20Template%20Method%20Pattern.md) |
| **11** | **Iterator** | Behavioral | Access elements of an aggregate object sequentially without exposing representation | [11 - Iterator Pattern.md](01-Design-Patterns/11-Iterator-Pattern/11%20-%20Iterator%20Pattern.md) |
| **12** | **Composite** | Structural | Compose objects into tree structures to represent part-whole hierarchies uniformly | [12 - Composite Pattern.md](01-Design-Patterns/12-Composite-Pattern/12%20-%20Composite%20Pattern.md) |
| **13** | **State** | Behavioral | Allow an object to alter behavior when internal state changes (clean state machine) | [13 - State Pattern.md](01-Design-Patterns/13-State-Pattern/13%20-%20State%20Pattern.md) |
| **14** | **Proxy** | Structural | Provide a surrogate or placeholder for another object to control access, caching, or lazy loading | [14 - Proxy Pattern.md](01-Design-Patterns/14-Proxy-Pattern/14%20-%20Proxy%20Pattern.md) |

---

## 🎯 Part 2: Top 50 LLD Interview Questions

The top 50 low-level design problems asked at Google, Meta, Amazon, Microsoft, Uber, and Stripe. Every question follows the **5-Step Interview Framework**: Requirements $\to$ Class Design & UML $\to$ Clean Code Implementation $\to$ Concurrency & Thread Safety $\to$ Follow-ups.

| # | Question Title | Difficulty | Key Patterns Applied | Architecture & Code |
|:---:|:---|:---:|:---|:---|
| **01** | **Parking Lot** | 🟢 Easy | Strategy, Factory | [Design a Parking Lot](02-LLD-Interview-Questions/01-Parking-Lot/Design%20a%20Parking%20Lot.md) |
| **02** | **Elevator System** | 🟡 Medium | State, Strategy, Observer | [Design an Elevator System](02-LLD-Interview-Questions/02-Elevator-System/Design%20an%20Elevator%20System.md) |
| **03** | **Library Management System** | 🟢 Easy | Strategy, Observer | [Design a Library Management System](02-LLD-Interview-Questions/03-Library-Management/Design%20a%20Library%20Management%20System.md) |
| **04** | **Vending Machine** | 🟢 Easy | State, Strategy | [Design a Vending Machine](02-LLD-Interview-Questions/04-Vending-Machine/Design%20a%20Vending%20Machine.md) |
| **05** | **Tic-Tac-Toe Game** | 🟢 Easy | Strategy | [Design a Tic-Tac-Toe Game](02-LLD-Interview-Questions/05-Tic-Tac-Toe/Design%20a%20Tic-Tac-Toe%20Game.md) |
| **06** | **Chess Game** | 🔴 Hard | Strategy, State, Command | [Design a Chess Game](02-LLD-Interview-Questions/06-Chess-Game/Design%20a%20Chess%20Game.md) |
| **07** | **Hotel Booking System** | 🟡 Medium | Strategy, Observer | [Design a Hotel Booking System](02-LLD-Interview-Questions/07-Hotel-Booking/Design%20a%20Hotel%20Booking%20System.md) |
| **08** | **Movie Ticket Booking System** | 🟡 Medium | Strategy, Observer | [Design a Movie Ticket Booking System](02-LLD-Interview-Questions/08-Movie-Ticket-Booking/Design%20a%20Movie%20Ticket%20Booking%20System.md) |
| **09** | **ATM System** | 🟡 Medium | State, Command | [Design an ATM System](02-LLD-Interview-Questions/09-ATM-System/Design%20an%20ATM%20System.md) |
| **10** | **Car Rental System** | 🟡 Medium | Strategy, Factory, Decorator | [Design a Car Rental System](02-LLD-Interview-Questions/10-Car-Rental/Design%20a%20Car%20Rental%20System.md) |
| **11** | **Social Media Feed** | 🟡 Medium | Observer, Iterator, Strategy | [Design a Social Media Feed](02-LLD-Interview-Questions/11-Social-Media-Feed/Design%20a%20Social%20Media%20Feed.md) |
| **12** | **Notification Service** | 🟡 Medium | Observer, Strategy, Factory | [Design a Notification Service (LLD)](02-LLD-Interview-Questions/12-Notification-Service/Design%20a%20Notification%20Service%20%28LLD%29.md) |
| **13** | **Online Shopping Cart** | 🟡 Medium | Strategy, Decorator | [Design an Online Shopping Cart](02-LLD-Interview-Questions/13-Shopping-Cart/Design%20an%20Online%20Shopping%20Cart.md) |
| **14** | **File System** | 🟡 Medium | Composite, Iterator | [Design a File System](02-LLD-Interview-Questions/14-File-System/Design%20a%20File%20System.md) |
| **15** | **Logging Framework** | 🟡 Medium | Singleton, Strategy, Decorator | [Design a Logging Framework](02-LLD-Interview-Questions/15-Logging-Framework/Design%20a%20Logging%20Framework.md) |
| **16** | **Cache (LRU / LFU)** | 🟡 Medium | Strategy, Proxy | [Design a Cache](02-LLD-Interview-Questions/16-Cache/Design%20a%20Cache.md) |
| **17** | **Pub-Sub Messaging System** | 🟡 Medium | Observer, Strategy | [Design a Pub-Sub Messaging System](02-LLD-Interview-Questions/17-Pub-Sub/Design%20a%20Pub-Sub%20Messaging%20System.md) |
| **18** | **Task Scheduler** | 🟡 Medium | Command, Strategy | [Design a Task Scheduler](02-LLD-Interview-Questions/18-Task-Scheduler/Design%20a%20Task%20Scheduler.md) |
| **19** | **Snake and Ladder Game** | 🟢 Easy | State, Strategy | [Design a Snake and Ladder Game](02-LLD-Interview-Questions/19-Snake-Ladder/Design%20a%20Snake%20and%20Ladder%20Game.md) |
| **20** | **Splitwise (Expense Sharing)** | 🔴 Hard | Strategy, Observer | [Design Splitwise](02-LLD-Interview-Questions/20-Splitwise/Design%20Splitwise.md) |
| **21** | **Stack Overflow (Q&A)** | 🟡 Medium | Observer, Strategy | [Design Stack Overflow](02-LLD-Interview-Questions/21-Stack-Overflow/Design%20Stack%20Overflow.md) |
| **22** | **Distributed Rate Limiter (LLD)** | 🟡 Medium | Strategy | [Design a Rate Limiter](02-LLD-Interview-Questions/22-Rate-Limiter/Design%20a%20Rate%20Limiter.md) |
| **23** | **URL Shortener (LLD)** | 🟡 Medium | Strategy | [Design a URL Shortener (LLD)](02-LLD-Interview-Questions/23-URL-Shortener/Design%20a%20URL%20Shortener%20%28LLD%29.md) |
| **24** | **Payment Processing System** | 🔴 Hard | Strategy, State, Command | [Design a Payment Processing System](02-LLD-Interview-Questions/24-Payment-System/Design%20a%20Payment%20Processing%20System.md) |
| **25** | **Food Delivery System** | 🟡 Medium | Strategy, State, Observer | [Design a Food Delivery System](02-LLD-Interview-Questions/25-Food-Delivery/Design%20a%20Food%20Delivery%20System.md) |
| **26** | **Ride-Sharing System (Uber/Lyft)** | 🟡 Medium | Strategy, State, Observer | [Design a Ride-Sharing System](02-LLD-Interview-Questions/26-Ride-Sharing/Design%20a%20Ride-Sharing%20System.md) |
| **27** | **Chat Application (WhatsApp/Slack)** | 🔴 Hard | Observer, Command, Mediator | [Design a Chat Application](02-LLD-Interview-Questions/27-Chat-Application/Design%20a%20Chat%20Application.md) |
| **28** | **Amazon Locker System** | 🟡 Medium | State, Strategy, Observer | [Design Amazon Locker System](02-LLD-Interview-Questions/28-Amazon-Locker/Design%20Amazon%20Locker%20System.md) |
| **29** | **HashMap from Scratch** | 🟢 Easy | Hash Table Bucketing | [Design a HashMap](02-LLD-Interview-Questions/29-HashMap/Design%20a%20HashMap.md) |
| **30** | **Inventory Management System** | 🟡 Medium | Observer, Strategy | [Design an Inventory Management System](02-LLD-Interview-Questions/30-Inventory-Management/Design%20an%20Inventory%20Management%20System.md) |
| **31** | **Spreadsheet Application (Excel)** | 🔴 Hard | Observer, Composite, Command | [Design a Spreadsheet](02-LLD-Interview-Questions/31-Spreadsheet/Design%20a%20Spreadsheet.md) |
| **32** | **Airline Booking System** | 🟡 Medium | State, Strategy | [Design an Airline Booking System](02-LLD-Interview-Questions/32-Airline-Booking/Design%20an%20Airline%20Booking%20System.md) |
| **33** | **Music Streaming Service (Spotify)** | 🟡 Medium | State, Observer, Iterator | [Design a Music Streaming Service](02-LLD-Interview-Questions/33-Music-Streaming/Design%20a%20Music%20Streaming%20Service.md) |
| **34** | **Text Editor with Undo / Redo** | 🟡 Medium | Command, Memento | [Design a Text Editor](02-LLD-Interview-Questions/34-Text-Editor/Design%20a%20Text%20Editor.md) |
| **35** | **Database Connection Pool** | 🟡 Medium | Singleton, Proxy | [Design a Database Connection Pool](02-LLD-Interview-Questions/35-Connection-Pool/Design%20a%20Database%20Connection%20Pool.md) |
| **36** | **LinkedIn Professional Network** | 🟡 Medium | Observer, Strategy, State | [Design LinkedIn](02-LLD-Interview-Questions/36-LinkedIn/Design%20LinkedIn.md) |
| **37** | **Traffic Light System** | 🟢 Easy | State, Observer | [Design a Traffic Light System](02-LLD-Interview-Questions/37-Traffic-Light/Design%20a%20Traffic%20Light%20System.md) |
| **38** | **Online Auction System (eBay)** | 🔴 Hard | Observer, State, Strategy | [Design an Online Auction System](02-LLD-Interview-Questions/38-Online-Auction/Design%20an%20Online%20Auction%20System.md) |
| **39** | **Thread Pool Execution Engine** | 🟡 Medium | Command, Strategy | [Design a Thread Pool](02-LLD-Interview-Questions/39-Thread-Pool/Design%20a%20Thread%20Pool.md) |
| **40** | **Deck of Cards & Blackjack** | 🟢 Easy | Factory, Strategy | [Design a Deck of Cards](02-LLD-Interview-Questions/40-Deck-of-Cards/Design%20a%20Deck%20of%20Cards.md) |
| **41** | **Meeting Scheduler (Google Cal)** | 🟡 Medium | Observer, Strategy | [Design a Meeting Scheduler](02-LLD-Interview-Questions/41-Meeting-Scheduler/Design%20a%20Meeting%20Scheduler.md) |
| **42** | **Bowling Alley Scoring Engine** | 🟡 Medium | Strategy | [Design a Bowling Alley](02-LLD-Interview-Questions/42-Bowling-Alley/Design%20a%20Bowling%20Alley.md) |
| **43** | **Order Management System** | 🟡 Medium | State, Command, Observer | [Design an Order Management System](02-LLD-Interview-Questions/43-Order-Management/Design%20an%20Order%20Management%20System.md) |
| **44** | **API Gateway (Edge Routing)** | 🟡 Medium | Proxy, Decorator, Strategy | [Design an API Gateway](02-LLD-Interview-Questions/44-API-Gateway/Design%20an%20API%20Gateway.md) |
| **45** | **Todo List Application** | 🟡 Medium | Command, Composite | [Design a Todo List Application](02-LLD-Interview-Questions/45-Todo-List/Design%20a%20Todo%20List%20Application.md) |
| **46** | **Stock Exchange & Matching Engine**| 🔴 Hard | Observer, Strategy, Command | [Design a Stock Exchange System](02-LLD-Interview-Questions/46-Stock-Exchange/Design%20a%20Stock%20Exchange%20System.md) |
| **47** | **Key-Value Store (In-Memory)** | 🟢 Easy | Singleton, Observer | [Design a Key-Value Store](02-LLD-Interview-Questions/47-Key-Value-Store/Design%20a%20Key-Value%20Store.md) |
| **48** | **Event Bus Engine** | 🟡 Medium | Observer, Mediator | [Design an Event Bus](02-LLD-Interview-Questions/48-Event-Bus/Design%20an%20Event%20Bus.md) |
| **49** | **Database Query Builder** | 🔴 Hard | Builder, Composite, Strategy | [Design a Database Query Builder](02-LLD-Interview-Questions/49-Query-Builder/Design%20a%20Database%20Query%20Builder.md) |
| **50** | **Circuit Breaker (Resilience)** | 🔴 Hard | State, Proxy, Decorator | [Design a Circuit Breaker](02-LLD-Interview-Questions/50-Circuit-Breaker/Design%20a%20Circuit%20Breaker.md) |

---

## 🧭 Curated Interview Study Tracks

### 🟢 Track A: The 7-Day Core Foundations (Junior / SDE-1)
Essential object-oriented modeling, encapsulation, and foundational state handling:
1. [Design a Parking Lot](02-LLD-Interview-Questions/01-Parking-Lot/Design%20a%20Parking%20Lot.md)
2. [Design a Tic-Tac-Toe Game](02-LLD-Interview-Questions/05-Tic-Tac-Toe/Design%20a%20Tic-Tac-Toe%20Game.md)
3. [Design a Vending Machine](02-LLD-Interview-Questions/04-Vending-Machine/Design%20a%20Vending%20Machine.md)
4. [Design a Library Management System](02-LLD-Interview-Questions/03-Library-Management/Design%20a%20Library%20Management%20System.md)
5. [Design a HashMap from Scratch](02-LLD-Interview-Questions/29-HashMap/Design%20a%20HashMap.md)
6. [Design a Deck of Cards](02-LLD-Interview-Questions/40-Deck-of-Cards/Design%20a%20Deck%20of%20Cards.md)
7. [Design an In-Memory Key-Value Store](02-LLD-Interview-Questions/47-Key-Value-Store/Design%20a%20Key-Value%20Store.md)

### 🟡 Track B: The 14-Day FAANG SDE-2 High-Probability Track
Real-time state transitions, thread safety, and multi-component coordination:
1. [Design an Elevator System](02-LLD-Interview-Questions/02-Elevator-System/Design%20an%20Elevator%20System.md)
2. [Design an LRU/LFU Cache](02-LLD-Interview-Questions/16-Cache/Design%20a%20Cache.md)
3. [Design a Movie Ticket Booking System](02-LLD-Interview-Questions/08-Movie-Ticket-Booking/Design%20a%20Movie%20Ticket%20Booking%20System.md)
4. [Design an ATM System](02-LLD-Interview-Questions/09-ATM-System/Design%20an%20ATM%20System.md)
5. [Design a Pub-Sub Messaging System](02-LLD-Interview-Questions/17-Pub-Sub/Design%20a%20Pub-Sub%20Messaging%20System.md)
6. [Design a Task Scheduler](02-LLD-Interview-Questions/18-Task-Scheduler/Design%20a%20Task%20Scheduler.md)
7. [Design a Food Delivery System (DoorDash)](02-LLD-Interview-Questions/25-Food-Delivery/Design%20a%20Food%20Delivery%20System.md)
8. [Design a Ride-Sharing System (Uber)](02-LLD-Interview-Questions/26-Ride-Sharing/Design%20a%20Ride-Sharing%20System.md)
9. [Design a Rate Limiter](02-LLD-Interview-Questions/22-Rate-Limiter/Design%20a%20Rate%20Limiter.md)
10. [Design a Database Connection Pool](02-LLD-Interview-Questions/35-Connection-Pool/Design%20a%20Database%20Connection%20Pool.md)

### 🔴 Track C: The Staff / Principal (L6+) Advanced Concurrency Track
High-throughput matching engines, DAG evaluation, and fault-tolerant resilient primitives:
1. [Design a Stock Exchange & Matching Engine](02-LLD-Interview-Questions/46-Stock-Exchange/Design%20a%20Stock%20Exchange%20System.md)
2. [Design a Resilient Circuit Breaker](02-LLD-Interview-Questions/50-Circuit-Breaker/Design%20a%20Circuit%20Breaker.md)
3. [Design a Spreadsheet Engine with Dependency Graphs](02-LLD-Interview-Questions/31-Spreadsheet/Design%20a%20Spreadsheet.md)
4. [Design Splitwise with Graph Debt Simplification](02-LLD-Interview-Questions/20-Splitwise/Design%20Splitwise.md)
5. [Design a Payment Processing System with Sagas](02-LLD-Interview-Questions/24-Payment-System/Design%20a%20Payment%20Processing%20System.md)
6. [Design an Event Bus with Dynamic Topic Filtering](02-LLD-Interview-Questions/48-Event-Bus/Design%20an%20Event%20Bus.md)
7. [Design a Thread Pool Execution Engine](02-LLD-Interview-Questions/39-Thread-Pool/Design%20a%20Thread%20Pool.md)

---

## 🎙️ The 5-Step LLD Interview Execution Framework (45 Minutes)

```
00:00 ────── 05:00 ────── 12:00 ────────────────── 32:00 ───────────── 40:00 ───── 45:00
  │            │            │                         │                   │          │
Clarify      Entities    Class Diagram &           Write Clean         Concurrency  Follow-ups
Scope        & Models    Patterns Applied          Code & Logic        & Safety     & Scale
```

1. **Phase 1: Clarify Scope & Invariants (Minutes 0:00 – 0:05)**:
   - Identify core actors, primary use cases, and non-functional constraints (concurrency, idempotency, latency).
2. **Phase 2: Identify Core Entities & Signatures (Minutes 0:05 – 0:12)**:
   - Extract nouns $\to$ Classes; verbs $\to$ Methods; adjectives/states $\to$ Enums.
3. **Phase 3: Class Diagram & Pattern Selection (Minutes 0:12 – 0:20)**:
   - Map relationships: Composition (`has-a`), Inheritance (`is-a`), Dependency (`uses-a`).
   - Select design patterns to decouple changing axes (Strategy for algorithms, State for lifecycles, Observer for notifications).
4. **Phase 4: Write Working, Extensible Code (Minutes 0:20 – 0:35)**:
   - Implement business logic adhering to SOLID principles. Write clean interfaces first, concrete implementations second.
5. **Phase 5: Concurrency, Thread Safety & Edge Cases (Minutes 0:35 – 0:45)**:
   - Defend against race conditions (Double-checked locks, mutexes, atomic variables). Explain extensibility trade-offs.

---

## 🏛️ SOLID Principles Quick Reference

| Principle | Core Definition | Violation Smell | Architectural Solution |
|:---:|:---|:---|:---|
| **S** | **Single Responsibility Principle** | Class does database access, JSON formatting, and billing | Split into repository, serializer, and billing domain services |
| **O** | **Open/Closed Principle** | `switch` or `if/else` ladders checked on every new payment type | Strategy Pattern: inject new strategy without modifying caller |
| **L** | **Liskov Substitution Principle** | Subclass overrides base method with `throw UnsupportedOperationException` | Segregate interfaces; favor composition over rigid inheritance |
| **I** | **Interface Segregation Principle** | Interface forces dummy implementations of unused methods | Break fat interfaces into cohesive, focused interfaces |
| **D** | **Dependency Inversion Principle** | High-level business logic directly instantiates concrete SQL driver | Inject abstract interfaces (`IDatabase`), decoupled from drivers |

---

## 📜 License & Acknowledgments

Curated and structured for software engineers preparing for L4, L5, and L6 technical interviews. Grounded in the teachings of *Head First Design Patterns*, *Design Patterns: Elements of Reusable Object-Oriented Software (GoF)*, and production clean-code standards.
