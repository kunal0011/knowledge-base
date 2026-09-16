---
date: "2026-04-06"
type: lld-question
difficulty: medium
status: active
tags: [lld, interview-prep, movie-ticket, booking]
---

# Design a Movie Ticket Booking System

## 1. Problem Statement
Design a movie ticket booking system (like BookMyShow) supporting show browsing, seat selection, booking, and payment.

## 2. Requirements
| # | Requirement |
|---|-------------|
| FR1 | Browse movies and shows by city, theater, time |
| FR2 | View seat map and select seats |
| FR3 | Temporary seat locking during booking (5 min) |
| FR4 | Process payment and confirm booking |
| FR5 | Cancel booking with refund policy |

## 3. Class Design

```mermaid
classDiagram
    class Movie {
        -String title
        -int duration
        -String genre
    }
    class Theater {
        -String name
        -String city
        -List~Screen~ screens
    }
    class Screen {
        -int screenNumber
        -int totalSeats
        -List~Show~ shows
    }
    class Show {
        -Movie movie
        -DateTime startTime
        -Screen screen
        -Map~Seat,SeatStatus~ seatMap
        +getAvailableSeats() List~Seat~
        +lockSeats(seats, userId) bool
        +bookSeats(seats, userId) Booking
    }
    class Seat {
        -String seatId
        -int row
        -int col
        -SeatType type
        -double price
    }
    class Booking {
        -String bookingId
        -User user
        -Show show
        -List~Seat~ seats
        -double totalAmount
        -BookingStatus status
    }
    class SeatType {
        <<enumeration>>
        REGULAR
        PREMIUM
        VIP
    }

    Theater *-- Screen
    Screen *-- Show
    Show --> Movie
    Show --> Seat
    Booking --> Show
    Booking --> Seat
```

### Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    actor Customer
    participant App as BookingApp
    participant Show as ShowService
    participant Lock as SeatLockManager
    participant Payment as PaymentGateway

    Customer->>App: selectSeats(showId, seatIds=["A1", "A2"])
    App->>Show: lockSeats(seatIds, userId)
    Show->>Lock: acquireLock(seatIds, duration=5min)
    Lock-->>Show: Lock Granted (Status: LOCKED)
    Show-->>App: Seats Locked (Start 5-min timer)
    App->>Payment: pay(amount=$30.00)
    Payment-->>App: Payment Success
    App->>Show: confirmBooking(seatIds, paymentId)
    Show->>Lock: commitBooking(seatIds)
    Lock-->>Show: Seats Status -> BOOKED
    Show-->>Customer: Tickets Issued
```


## 4. Key Implementation (Python)

```python
from enum import Enum
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import uuid, threading

class SeatStatus(Enum):
    AVAILABLE = "AVAILABLE"
    LOCKED = "LOCKED"
    BOOKED = "BOOKED"

class Seat:
    def __init__(self, seat_id: str, row: int, col: int, price: float):
        self.seat_id = seat_id
        self.row = row
        self.col = col
        self.price = price

class Show:
    LOCK_DURATION = timedelta(minutes=5)

    def __init__(self, movie_title: str, start_time: datetime, seats: List[Seat]):
        self.movie_title = movie_title
        self.start_time = start_time
        self._seat_status: Dict[str, SeatStatus] = {s.seat_id: SeatStatus.AVAILABLE for s in seats}
        self._seat_map: Dict[str, Seat] = {s.seat_id: s for s in seats}
        self._locks: Dict[str, datetime] = {}  # seat_id → lock expiry
        self._lock = threading.Lock()

    def get_available_seats(self) -> List[Seat]:
        self._release_expired_locks()
        return [self._seat_map[sid] for sid, status in self._seat_status.items()
                if status == SeatStatus.AVAILABLE]

    def lock_seats(self, seat_ids: List[str], user_id: str) -> bool:
        with self._lock:
            self._release_expired_locks()
            # Check all seats available
            for sid in seat_ids:
                if self._seat_status.get(sid) != SeatStatus.AVAILABLE:
                    return False
            # Lock all
            expiry = datetime.now() + self.LOCK_DURATION
            for sid in seat_ids:
                self._seat_status[sid] = SeatStatus.LOCKED
                self._locks[sid] = expiry
            return True

    def book_seats(self, seat_ids: List[str]) -> bool:
        with self._lock:
            for sid in seat_ids:
                if self._seat_status.get(sid) != SeatStatus.LOCKED:
                    return False
            for sid in seat_ids:
                self._seat_status[sid] = SeatStatus.BOOKED
                self._locks.pop(sid, None)
            return True

    def _release_expired_locks(self):
        now = datetime.now()
        expired = [sid for sid, exp in self._locks.items() if now > exp]
        for sid in expired:
            self._seat_status[sid] = SeatStatus.AVAILABLE
            del self._locks[sid]

class BookingService:
    def __init__(self):
        self.bookings: Dict[str, dict] = {}

    def create_booking(self, user_id: str, show: Show,
                       seat_ids: List[str]) -> Optional[str]:
        # Step 1: Lock seats
        if not show.lock_seats(seat_ids, user_id):
            print("❌ Seats not available")
            return None

        # Step 2: (Payment would happen here)
        # Step 3: Confirm booking
        if show.book_seats(seat_ids):
            booking_id = str(uuid.uuid4())[:8]
            total = sum(show._seat_map[sid].price for sid in seat_ids)
            self.bookings[booking_id] = {
                "user_id": user_id, "seats": seat_ids, "total": total
            }
            print(f"✅ Booking {booking_id}: {len(seat_ids)} seats, ${total:.2f}")
            return booking_id
        return None
```

### Java

```java
package com.lld.movieticket;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

enum SeatStatus { AVAILABLE, LOCKED, BOOKED }

class Seat {
    private final String seatId;
    private final int row;
    private final int col;
    private final double price;

    public Seat(String seatId, int row, int col, double price) {
        this.seatId = seatId;
        this.row = row;
        this.col = col;
        this.price = price;
    }
    public String getSeatId() { return seatId; }
    public double getPrice() { return price; }
}

public class MovieTicketBookingSystem {
    private final Map<String, SeatStatus> seatStatus = new ConcurrentHashMap<>();
    private final Map<String, LocalDateTime> lockExpirations = new ConcurrentHashMap<>();
    private final Map<String, Seat> seats = new HashMap<>();
    private final ReentrantLock lock = new ReentrantLock();
    private static final int LOCK_DURATION_MINUTES = 5;

    public void addSeat(Seat seat) {
        seats.put(seat.getSeatId(), seat);
        seatStatus.put(seat.getSeatId(), SeatStatus.AVAILABLE);
    }

    public boolean lockSeats(List<String> seatIds, String userId) {
        lock.lock();
        try {
            cleanupExpiredLocks();
            // Validate all seats are available
            for (String sid : seatIds) {
                if (seatStatus.get(sid) != SeatStatus.AVAILABLE) {
                    return false;
                }
            }
            // Acquire temporary lock
            LocalDateTime expiry = LocalDateTime.now().plusMinutes(LOCK_DURATION_MINUTES);
            for (String sid : seatIds) {
                seatStatus.put(sid, SeatStatus.LOCKED);
                lockExpirations.put(sid, expiry);
            }
            return true;
        } finally {
            lock.unlock();
        }
    }

    public boolean bookSeats(List<String> seatIds, String userId) {
        lock.lock();
        try {
            for (String sid : seatIds) {
                if (seatStatus.get(sid) != SeatStatus.LOCKED) {
                    return false;
                }
            }
            for (String sid : seatIds) {
                seatStatus.put(sid, SeatStatus.BOOKED);
                lockExpirations.remove(sid);
            }
            return true;
        } finally {
            lock.unlock();
        }
    }

    private void cleanupExpiredLocks() {
        LocalDateTime now = LocalDateTime.now();
        for (Map.Entry<String, LocalDateTime> entry : lockExpirations.entrySet()) {
            if (now.isAfter(entry.getValue())) {
                seatStatus.put(entry.getKey(), SeatStatus.AVAILABLE);
                lockExpirations.remove(entry.getKey());
            }
        }
    }
}
```



---

## Thread Safety Considerations

| Concern | Solution |
|---|---|
| Concurrent seat locking race | `ReentrantLock` guards `lockSeats()` so batch checks and lock allocations are fully atomic |
| Abandoned carts / Lock expiry | Lazy lock cleanup during retrieval + scheduled background reaper thread |

## Extensibility & SOLID Principles

| Principle | Architectural Implementation |
|---|---|
| **S** — Single Responsibility | `SeatLockManager` separates ephemeral locking logic from `Show` inventory and `Payment` services |
| **O** — Open/Closed | Pricing strategies (VIP, matinee, surge) easily injected via Strategy pattern |
| **D** — Dependency Inversion | Payment processor decoupled behind interface abstraction |

---

## 5. Key Design Decisions

> [!important] Seat Locking
> Temporary seat locking (5 min) is critical to prevent double-booking. This is the most asked follow-up. Use optimistic locking with TTL or distributed locks (Redis SETNX).

## 6. Follow-ups
- **How to handle concurrent bookings?** Distributed locks (Redis), optimistic locking with version numbers.
- **How to handle partial failures?** Saga pattern — if payment fails, release locks.
- **How to handle seat map display?** WebSocket for real-time seat availability updates.

---

**Related:** [[01 - Strategy Pattern]] | [[02 - Observer Pattern]]
