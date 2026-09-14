---
date: "2026-04-06"
type: project
status: active
tags:
  - lld
  - design-patterns
  - interview-prep
  - object-oriented-design
---

# LLD — Design Patterns & Interview Questions

> [!note] About
> Comprehensive low-level design resource covering all **14 design patterns from Head First Design Patterns** and the **top 50 LLD interview questions**. Each pattern includes problem statements, key principles, UML diagrams, and full Python & Java implementations. Each interview question follows a structured interview-format answer.

## Part 1: Head First Design Patterns

| # | Pattern | Category | Key Principle | Link |
|---|---------|----------|---------------|------|
| 1 | Strategy | Behavioral | Favor composition over inheritance | [[01 - Strategy Pattern]] |
| 2 | Observer | Behavioral | Loose coupling | [[02 - Observer Pattern]] |
| 3 | Decorator | Structural | Open-Closed Principle | [[03 - Decorator Pattern]] |
| 4 | Factory Method | Creational | Depend on abstractions | [[04 - Factory Method Pattern]] |
| 5 | Abstract Factory | Creational | Family of related objects | [[05 - Abstract Factory Pattern]] |
| 6 | Singleton | Creational | One instance, global access | [[06 - Singleton Pattern]] |
| 7 | Command | Behavioral | Encapsulate requests as objects | [[07 - Command Pattern]] |
| 8 | Adapter | Structural | Convert interfaces | [[08 - Adapter Pattern]] |
| 9 | Facade | Structural | Simplified interface | [[09 - Facade Pattern]] |
| 10 | Template Method | Behavioral | Algorithm skeleton | [[10 - Template Method Pattern]] |
| 11 | Iterator | Behavioral | Sequential access | [[11 - Iterator Pattern]] |
| 12 | Composite | Structural | Part-whole hierarchies | [[12 - Composite Pattern]] |
| 13 | State | Behavioral | State-based behavior | [[13 - State Pattern]] |
| 14 | Proxy | Structural | Surrogate / placeholder | [[14 - Proxy Pattern]] |

→ Full patterns index: [[Design Patterns - Index]]

## Part 2: Top 50 LLD Interview Questions

| # | Question | Difficulty | Key Patterns | Link |
|---|----------|-----------|--------------|------|
| 1 | Parking Lot | 🟢 Easy | Strategy, Factory | [[Design a Parking Lot]] |
| 2 | Elevator System | 🟡 Medium | State, Strategy, Observer | [[Design an Elevator System]] |
| 3 | Library Management System | 🟢 Easy | Strategy, Observer | [[Design a Library Management System]] |
| 4 | Vending Machine | 🟢 Easy | State, Strategy | [[Design a Vending Machine]] |
| 5 | Tic-Tac-Toe Game | 🟢 Easy | Strategy | [[Design a Tic-Tac-Toe Game]] |
| 6 | Chess Game | 🔴 Hard | Strategy, State, Command | [[Design a Chess Game]] |
| 7 | Hotel Booking System | 🟡 Medium | Strategy, Observer | [[Design a Hotel Booking System]] |
| 8 | Movie Ticket Booking System | 🟡 Medium | Strategy, Observer | [[Design a Movie Ticket Booking System]] |
| 9 | ATM System | 🟡 Medium | State, Command | [[Design an ATM System]] |
| 10 | Car Rental System | 🟡 Medium | Strategy, Factory, Decorator | [[Design a Car Rental System]] |
| 11 | Social Media Feed | 🟡 Medium | Observer, Iterator, Strategy | [[Design a Social Media Feed]] |
| 12 | Notification Service | 🟡 Medium | Observer, Strategy, Factory | [[Design a Notification Service (LLD)]] |
| 13 | Online Shopping Cart | 🟡 Medium | Strategy, Decorator | [[Design an Online Shopping Cart]] |
| 14 | File System | 🟡 Medium | Composite, Iterator | [[Design a File System]] |
| 15 | Logging Framework | 🟡 Medium | Singleton, Strategy, Decorator | [[Design a Logging Framework]] |
| 16 | Cache (LRU/LFU) | 🟡 Medium | Strategy, Proxy | [[Design a Cache]] |
| 17 | Pub-Sub Messaging System | 🟡 Medium | Observer, Strategy | [[Design a Pub-Sub Messaging System]] |
| 18 | Task Scheduler | 🟡 Medium | Command, Strategy | [[Design a Task Scheduler]] |
| 19 | Snake and Ladder Game | 🟢 Easy | State, Strategy | [[Design a Snake and Ladder Game]] |
| 20 | Splitwise (Expense Sharing) | 🔴 Hard | Strategy, Observer | [[Design Splitwise]] |
| 21 | Stack Overflow (Q&A) | 🟡 Medium | Observer, Strategy | [[Design Stack Overflow]] |
| 22 | Rate Limiter | 🟡 Medium | Strategy | [[Design a Rate Limiter]] |
| 23 | URL Shortener (LLD) | 🟡 Medium | Strategy | [[Design a URL Shortener (LLD)]] |
| 24 | Payment Processing System | 🔴 Hard | Strategy, State, Command | [[Design a Payment Processing System]] |
| 25 | Food Delivery System | 🟡 Medium | Strategy, State, Observer | [[Design a Food Delivery System]] |
| 26 | Ride-Sharing System | 🟡 Medium | Strategy, State, Observer | [[Design a Ride-Sharing System]] |
| 27 | Chat Application | 🔴 Hard | Observer, Command, Mediator | [[Design a Chat Application]] |
| 28 | Amazon Locker System | 🟡 Medium | State, Strategy, Observer | [[Design Amazon Locker System]] |
| 29 | HashMap from Scratch | 🟢 Easy | Hash Table Bucketing | [[Design a HashMap]] |
| 30 | Inventory Management System | 🟡 Medium | Observer, Strategy | [[Design an Inventory Management System]] |
| 31 | Spreadsheet Application | 🔴 Hard | Observer, Composite, Command | [[Design a Spreadsheet]] |
| 32 | Airline Booking System | 🟡 Medium | State, Strategy | [[Design an Airline Booking System]] |
| 33 | Music Streaming Service | 🟡 Medium | State, Observer, Iterator | [[Design a Music Streaming Service]] |
| 34 | Text Editor (Undo/Redo) | 🟡 Medium | Command, Memento | [[Design a Text Editor]] |
| 35 | Database Connection Pool | 🟡 Medium | Singleton, Proxy | [[Design a Database Connection Pool]] |
| 36 | LinkedIn Professional Network | 🟡 Medium | Observer, Strategy, State | [[Design LinkedIn]] |
| 37 | Traffic Light System | 🟢 Easy | State, Observer | [[Design a Traffic Light System]] |
| 38 | Online Auction System | 🔴 Hard | Observer, State, Strategy | [[Design an Online Auction System]] |
| 39 | Thread Pool | 🟡 Medium | Command, Strategy | [[Design a Thread Pool]] |
| 40 | Deck of Cards | 🟢 Easy | Factory, Strategy | [[Design a Deck of Cards]] |
| 41 | Meeting Scheduler | 🟡 Medium | Observer, Strategy | [[Design a Meeting Scheduler]] |
| 42 | Bowling Alley Scoring | 🟡 Medium | Strategy | [[Design a Bowling Alley]] |
| 43 | Order Management System | 🟡 Medium | State, Command, Observer | [[Design an Order Management System]] |
| 44 | API Gateway | 🟡 Medium | Proxy, Decorator, Strategy | [[Design an API Gateway]] |
| 45 | Todo List Application | 🟡 Medium | Command, Composite | [[Design a Todo List Application]] |
| 46 | Stock Exchange System | 🔴 Hard | Observer, Strategy, Command | [[Design a Stock Exchange System]] |
| 47 | Key-Value Store | 🟢 Easy | Singleton, Observer | [[Design a Key-Value Store]] |
| 48 | Event Bus | 🟡 Medium | Observer, Mediator | [[Design an Event Bus]] |
| 49 | Database Query Builder | 🔴 Hard | Builder, Composite, Strategy | [[Design a Database Query Builder]] |
| 50 | Circuit Breaker | 🔴 Hard | State, Proxy, Decorator | [[Design a Circuit Breaker]] |

→ Full questions index: [[LLD Interview Questions - Index]]

## Interview Framework — LLD

```
LLD Interview Flow (45 min):
├── 1. Requirements Clarification (3-5 min)
│   ├── Functional Requirements
│   ├── Non-Functional Requirements (concurrency, scale)
│   └── Constraints & Assumptions
│
├── 2. Identify Core Objects & Relationships (5-7 min)
│   ├── Nouns → Classes
│   ├── Verbs → Methods
│   └── Adjectives → Enums / States
│
├── 3. Class Diagram (10-12 min)
│   ├── Inheritance vs Composition
│   ├── Interfaces & Abstract Classes
│   └── Design Patterns to Apply
│
├── 4. Write Key Classes (15-20 min)
│   ├── Core business logic
│   ├── Thread safety considerations
│   └── SOLID principles
│
└── 5. Extensibility & Trade-offs (5 min)
    ├── How to add new features?
    ├── What would change if scale increases?
    └── Alternative design decisions
```

## SOLID Principles Quick Reference

| Principle | Meaning | Example |
|-----------|---------|---------|
| **S** — Single Responsibility | One class = one reason to change | Separate `PaymentProcessor` from `EmailNotifier` |
| **O** — Open/Closed | Open for extension, closed for modification | Strategy pattern for payment types |
| **L** — Liskov Substitution | Subtypes must be substitutable for base types | `Rectangle`/`Square` problem |
| **I** — Interface Segregation | Prefer small, focused interfaces | Split `IWorker` into `IWorkable` + `IEatable` |
| **D** — Dependency Inversion | Depend on abstractions, not concretions | Inject `IPaymentGateway`, not `StripeGateway` |

---

> [!tip] Dataview Query — All LLD Notes
> ```dataview
> TABLE tags, status
> FROM "Projects/LLD"
> WHERE type = "design-pattern" OR type = "lld-question"
> SORT file.name ASC
> ```
