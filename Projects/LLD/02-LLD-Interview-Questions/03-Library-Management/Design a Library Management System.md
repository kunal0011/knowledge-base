---
date: "2026-04-06"
type: lld-question
difficulty: easy
status: active
tags:
  - lld
  - interview-prep
  - library-management
---

# Design a Library Management System

## 1. Problem Statement

Design a system for a library to manage books, members, borrowing/returning, and reservations.

---

## 2. Requirements

### Functional Requirements

| # | Requirement |
|---|-------------|
| FR1 | Add/remove/search books by title, author, ISBN |
| FR2 | Register/manage library members |
| FR3 | Issue (checkout) and return books |
| FR4 | Reserve books that are currently checked out |
| FR5 | Track overdue books and calculate fines |
| FR6 | Limit number of books a member can borrow (max 5) |

---

## 3. Class Design

```mermaid
classDiagram
    class Library {
        -String name
        -Map~String, List~BookItem~~ bookCatalog
        -Map~String, Member~ members
        +searchByTitle(title) List~Book~
        +searchByAuthor(author) List~Book~
        +issueBook(memberId, isbn) bool
        +returnBook(memberId, isbn) double
    }

    class Book {
        -String isbn
        -String title
        -String author
        -String publisher
        -int totalCopies
    }

    class BookItem {
        -String barcode
        -BookStatus status
        -DateTime dueDate
        -Member issuedTo
        +checkout(Member)
        +returnBook()
    }

    class Member {
        -String memberId
        -String name
        -List~BookItem~ borrowedBooks
        -int maxBooks
        +canBorrow() bool
        +borrowBook(BookItem)
        +returnBook(BookItem)
    }

    class Reservation {
        -String memberId
        -String isbn
        -DateTime reservedDate
        -ReservationStatus status
    }

    class BookStatus {
        <<enumeration>>
        AVAILABLE
        CHECKED_OUT
        RESERVED
        LOST
    }

    class FineCalculator {
        <<interface>>
        +calculateFine(dueDate, returnDate)* double
    }

    Library *-- Book
    Book *-- BookItem
    Library --> Member
    BookItem --> BookStatus
    Member --> BookItem : borrows
    Library --> Reservation
    Library --> FineCalculator
```

### Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    actor Member
    participant Library as LibraryService
    participant Catalog as BookCatalog
    participant Item as BookItem
    participant Fine as FineCalculator

    Member->>Library: searchByTitle("Clean Architecture")
    Library->>Catalog: findBooks("Clean Architecture")
    Catalog-->>Library: List of matching Books
    Library-->>Member: Available Books & ISBNs
    Member->>Library: issueBook(memberId, isbn)
    Library->>Item: check status & checkout(memberId)
    Item-->>Library: success (due date = +14 days)
    Library-->>Member: Book Issued Successfully
    Note over Member,Library: 20 Days Pass (Book Overdue)
    Member->>Library: returnBook(memberId, barcode)
    Library->>Fine: calculateFine(dueDate, now)
    Fine-->>Library: Fine Amount ($6.00)
    Library->>Item: returnBook()
    Library-->>Member: Book Returned (Fine: $6.00)
```


---

## 4. Key Implementation (Python)

```python
from enum import Enum
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from collections import defaultdict


class BookStatus(Enum):
    AVAILABLE = "AVAILABLE"
    CHECKED_OUT = "CHECKED_OUT"
    RESERVED = "RESERVED"
    LOST = "LOST"


class Book:
    def __init__(self, isbn: str, title: str, author: str):
        self.isbn = isbn
        self.title = title
        self.author = author


class BookItem:
    def __init__(self, barcode: str, book: Book):
        self.barcode = barcode
        self.book = book
        self.status = BookStatus.AVAILABLE
        self.due_date: Optional[datetime] = None
        self.issued_to: Optional[str] = None

    def checkout(self, member_id: str, loan_days: int = 14):
        self.status = BookStatus.CHECKED_OUT
        self.issued_to = member_id
        self.due_date = datetime.now() + timedelta(days=loan_days)

    def return_book(self):
        self.status = BookStatus.AVAILABLE
        self.issued_to = None
        self.due_date = None

    def is_overdue(self) -> bool:
        return (self.status == BookStatus.CHECKED_OUT and
                self.due_date and datetime.now() > self.due_date)


class Member:
    MAX_BOOKS = 5

    def __init__(self, member_id: str, name: str):
        self.member_id = member_id
        self.name = name
        self.borrowed_books: List[BookItem] = []

    def can_borrow(self) -> bool:
        return len(self.borrowed_books) < self.MAX_BOOKS

    def has_overdue(self) -> bool:
        return any(b.is_overdue() for b in self.borrowed_books)


class Library:
    FINE_PER_DAY = 0.50

    def __init__(self, name: str):
        self.name = name
        self._books: Dict[str, Book] = {}            # isbn → Book
        self._items: Dict[str, List[BookItem]] = defaultdict(list)  # isbn → [BookItem]
        self._members: Dict[str, Member] = {}

    def add_book(self, book: Book, copies: int = 1):
        self._books[book.isbn] = book
        for i in range(copies):
            barcode = f"{book.isbn}-{len(self._items[book.isbn]) + 1}"
            self._items[book.isbn].append(BookItem(barcode, book))

    def register_member(self, member: Member):
        self._members[member.member_id] = member

    def search_by_title(self, title: str) -> List[Book]:
        return [b for b in self._books.values()
                if title.lower() in b.title.lower()]

    def issue_book(self, member_id: str, isbn: str) -> bool:
        member = self._members.get(member_id)
        if not member or not member.can_borrow() or member.has_overdue():
            return False

        for item in self._items.get(isbn, []):
            if item.status == BookStatus.AVAILABLE:
                item.checkout(member_id)
                member.borrowed_books.append(item)
                print(f"✅ '{item.book.title}' issued to {member.name}")
                return True
        print(f"❌ No available copies of ISBN {isbn}")
        return False

    def return_book(self, member_id: str, barcode: str) -> float:
        member = self._members.get(member_id)
        if not member:
            raise ValueError("Invalid member")

        for item in member.borrowed_books:
            if item.barcode == barcode:
                fine = 0.0
                if item.is_overdue():
                    overdue_days = (datetime.now() - item.due_date).days
                    fine = overdue_days * self.FINE_PER_DAY
                item.return_book()
                member.borrowed_books.remove(item)
                print(f"📚 '{item.book.title}' returned. Fine: ${fine:.2f}")
                return fine
        raise ValueError("Book not found in member's borrowed list")
```

### Java

```java
package com.lld.library;

import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

enum BookStatus {
    AVAILABLE, CHECKED_OUT, RESERVED, LOST
}

class Book {
    private final String isbn;
    private final String title;
    private final String author;

    public Book(String isbn, String title, String author) {
        this.isbn = isbn;
        this.title = title;
        this.author = author;
    }
    public String getIsbn() { return isbn; }
    public String getTitle() { return title; }
}

class BookItem {
    private final String barcode;
    private final Book book;
    private BookStatus status = BookStatus.AVAILABLE;
    private LocalDate dueDate;
    private String borrowedByMemberId;
    private final ReentrantLock lock = new ReentrantLock();

    public BookItem(String barcode, Book book) {
        this.barcode = barcode;
        this.book = book;
    }

    public boolean checkout(String memberId, int loanDays) {
        lock.lock();
        try {
            if (status != BookStatus.AVAILABLE) return false;
            this.status = BookStatus.CHECKED_OUT;
            this.borrowedByMemberId = memberId;
            this.dueDate = LocalDate.now().plusDays(loanDays);
            return true;
        } finally {
            lock.unlock();
        }
    }

    public void returnBook() {
        lock.lock();
        try {
            this.status = BookStatus.AVAILABLE;
            this.borrowedByMemberId = null;
            this.dueDate = null;
        } finally {
            lock.unlock();
        }
    }

    public boolean isOverdue() {
        return dueDate != null && LocalDate.now().isAfter(dueDate);
    }

    public LocalDate getDueDate() { return dueDate; }
    public String getBarcode() { return barcode; }
    public Book getBook() { return book; }
    public BookStatus getStatus() { return status; }
}

class Member {
    private final String memberId;
    private final String name;
    private final List<BookItem> borrowedBooks = new ArrayList<>();
    private static final int MAX_BOOKS = 5;

    public Member(String memberId, String name) {
        this.memberId = memberId;
        this.name = name;
    }

    public synchronized boolean canBorrow() {
        return borrowedBooks.size() < MAX_BOOKS;
    }

    public synchronized void addBorrowedBook(BookItem item) {
        borrowedBooks.add(item);
    }

    public synchronized void removeBorrowedBook(BookItem item) {
        borrowedBooks.remove(item);
    }

    public String getMemberId() { return memberId; }
    public String getName() { return name; }
    public List<BookItem> getBorrowedBooks() { return Collections.unmodifiableList(borrowedBooks); }
}

public class LibraryManagementSystem {
    private final Map<String, Book> booksByIsbn = new ConcurrentHashMap<>();
    private final Map<String, List<BookItem>> itemsByIsbn = new ConcurrentHashMap<>();
    private final Map<String, Member> members = new ConcurrentHashMap<>();
    private static final double DAILY_FINE = 1.0;

    public void addBook(Book book, int copies) {
        booksByIsbn.put(book.getIsbn(), book);
        itemsByIsbn.putIfAbsent(book.getIsbn(), new ArrayList<>());
        for (int i = 1; i <= copies; i++) {
            itemsByIsbn.get(book.getIsbn()).add(new BookItem(book.getIsbn() + "-" + i, book));
        }
    }

    public void registerMember(Member member) {
        members.put(member.getMemberId(), member);
    }

    public synchronized boolean issueBook(String memberId, String isbn) {
        Member member = members.get(memberId);
        if (member == null || !member.canBorrow()) return false;

        List<BookItem> items = itemsByIsbn.get(isbn);
        if (items == null) return false;

        for (BookItem item : items) {
            if (item.getStatus() == BookStatus.AVAILABLE && item.checkout(memberId, 14)) {
                member.addBorrowedBook(item);
                return true;
            }
        }
        return false;
    }

    public synchronized double returnBook(String memberId, String barcode, String isbn) {
        Member member = members.get(memberId);
        if (member == null) throw new IllegalArgumentException("Invalid member");

        for (BookItem item : member.getBorrowedBooks()) {
            if (item.getBarcode().equals(barcode)) {
                double fine = 0.0;
                if (item.isOverdue()) {
                    long days = ChronoUnit.DAYS.between(item.getDueDate(), LocalDate.now());
                    fine = days * DAILY_FINE;
                }
                item.returnBook();
                member.removeBorrowedBook(item);
                return fine;
            }
        }
        throw new IllegalStateException("Book not borrowed by member");
    }
}
```


---


---

## Thread Safety Considerations

| Concern | Solution |
|---|---|
| Concurrent checkout of same copy | Item-level `ReentrantLock` ensures exactly one thread changes `BookStatus.AVAILABLE` to `CHECKED_OUT` |
| Member quota race | Synchronized `canBorrow()` and `addBorrowedBook()` block concurrent over-limit checkouts |
| Catalog reads vs writes | `ConcurrentHashMap` allows lock-free search operations concurrent with inventory additions |

## Extensibility & SOLID Principles

| Principle | Architectural Implementation |
|---|---|
| **S** — Single Responsibility | `BookItem` manages physical barcode status; `Member` tracks borrowing quota; `Library` orchestrates |
| **O** — Open/Closed | Lending policies and fine strategies can be swapped via strategy interfaces without modifying core catalog |
| **L** — Liskov Substitution | `ReferenceBook` or `EBook` subclasses can substitute for `Book` without breaking checkout contracts |
| **I** — Interface Segregation | Search APIs segregated from checkout/transactional APIs |
| **D** — Dependency Inversion | Fine calculation relies on a pluggable `FineCalculator` contract |

---

## 5. Follow-up Questions

**Q: How to handle reservations?** Add a `ReservationQueue` per ISBN. When a book is returned, notify the first person in queue.

**Q: How to handle e-books?** `EBook extends Book` with `downloadLink`. No physical copies — unlimited checkouts with DRM.

**Q: How to handle notifications?** Observer pattern — `Member` subscribes to `Book`, notified when available.

---

**Related:** [[02 - Observer Pattern]] | [[01 - Strategy Pattern]]
