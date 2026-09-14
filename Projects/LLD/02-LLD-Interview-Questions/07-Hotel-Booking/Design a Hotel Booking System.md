---
date: "2026-04-06"
type: lld-question
difficulty: medium
status: active
tags: [lld, interview-prep, hotel-booking, strategy-pattern]
---

# Design a Hotel Booking System

## 1. Problem Statement
Design a hotel booking system supporting room search, reservation, check-in/check-out, and payment.

## 2. Requirements
| # | Requirement |
|---|-------------|
| FR1 | Search available rooms by date range, room type, price |
| FR2 | Book a room for a date range |
| FR3 | Cancel booking |
| FR4 | Check-in and check-out |
| FR5 | Support multiple room types (single, double, suite) |
| FR6 | Handle payments and invoicing |

## 3. Class Design

```mermaid
classDiagram
    class Hotel {
        -String name
        -List~Room~ rooms
        +searchRooms(dateRange, roomType) List~Room~
        +bookRoom(guest, room, dateRange) Booking
    }
    class Room {
        -String roomNumber
        -RoomType type
        -double pricePerNight
        -RoomStatus status
        +isAvailable(dateRange) bool
    }
    class Booking {
        -String bookingId
        -Guest guest
        -Room room
        -DateRange dateRange
        -BookingStatus status
        -double totalAmount
        +cancel()
        +checkIn()
        +checkOut()
    }
    class Guest {
        -String guestId
        -String name
        -String email
    }
    class RoomType {
        <<enumeration>>
        SINGLE
        DOUBLE
        DELUXE
        SUITE
    }
    class BookingStatus {
        <<enumeration>>
        CONFIRMED
        CHECKED_IN
        CHECKED_OUT
        CANCELLED
    }
    class PaymentProcessor {
        <<interface>>
        +processPayment(amount, method)* bool
    }

    Hotel *-- Room
    Booking --> Room
    Booking --> Guest
    Booking --> BookingStatus
    Room --> RoomType
    Hotel --> PaymentProcessor
```

## 4. Key Implementation (Python)

```python
from enum import Enum
from datetime import date
from typing import List, Optional, Dict, Tuple
import uuid

class RoomType(Enum):
    SINGLE = ("Single", 100)
    DOUBLE = ("Double", 150)
    DELUXE = ("Deluxe", 250)
    SUITE = ("Suite", 400)

class BookingStatus(Enum):
    CONFIRMED = "CONFIRMED"
    CHECKED_IN = "CHECKED_IN"
    CHECKED_OUT = "CHECKED_OUT"
    CANCELLED = "CANCELLED"

class DateRange:
    def __init__(self, check_in: date, check_out: date):
        self.check_in = check_in
        self.check_out = check_out

    def overlaps(self, other: 'DateRange') -> bool:
        return self.check_in < other.check_out and other.check_in < self.check_out

    @property
    def nights(self) -> int:
        return (self.check_out - self.check_in).days

class Guest:
    def __init__(self, guest_id: str, name: str, email: str):
        self.guest_id = guest_id
        self.name = name
        self.email = email

class Room:
    def __init__(self, room_number: str, room_type: RoomType):
        self.room_number = room_number
        self.room_type = room_type
        self.price_per_night = room_type.value[1]
        self._bookings: List['Booking'] = []

    def is_available(self, date_range: DateRange) -> bool:
        return not any(
            b.date_range.overlaps(date_range)
            for b in self._bookings
            if b.status in (BookingStatus.CONFIRMED, BookingStatus.CHECKED_IN)
        )

class Booking:
    def __init__(self, guest: Guest, room: Room, date_range: DateRange):
        self.booking_id = str(uuid.uuid4())[:8]
        self.guest = guest
        self.room = room
        self.date_range = date_range
        self.status = BookingStatus.CONFIRMED
        self.total_amount = room.price_per_night * date_range.nights

    def cancel(self):
        if self.status == BookingStatus.CONFIRMED:
            self.status = BookingStatus.CANCELLED
            print(f"Booking {self.booking_id} cancelled")

    def check_in(self):
        if self.status == BookingStatus.CONFIRMED:
            self.status = BookingStatus.CHECKED_IN
            print(f"✅ Checked in: Room {self.room.room_number}")

    def check_out(self):
        if self.status == BookingStatus.CHECKED_IN:
            self.status = BookingStatus.CHECKED_OUT
            print(f"🚪 Checked out: Room {self.room.room_number}. Total: ${self.total_amount}")

class Hotel:
    def __init__(self, name: str):
        self.name = name
        self.rooms: List[Room] = []
        self.bookings: Dict[str, Booking] = {}

    def add_room(self, room: Room):
        self.rooms.append(room)

    def search_rooms(self, date_range: DateRange,
                     room_type: Optional[RoomType] = None) -> List[Room]:
        return [r for r in self.rooms
                if r.is_available(date_range) and
                (room_type is None or r.room_type == room_type)]

    def book_room(self, guest: Guest, room: Room, date_range: DateRange) -> Optional[Booking]:
        if not room.is_available(date_range):
            print("Room not available")
            return None
        booking = Booking(guest, room, date_range)
        room._bookings.append(booking)
        self.bookings[booking.booking_id] = booking
        print(f"✅ Booked room {room.room_number} for {guest.name}: "
              f"${booking.total_amount} ({date_range.nights} nights)")
        return booking
```

## 5. Follow-ups
- **Dynamic pricing?** Strategy pattern — `PricingStrategy` (peak, off-peak, holiday).
- **Room service?** Observer pattern — notify housekeeping on checkout.
- **Multi-hotel chain?** Add `HotelChain` with hotel catalog and cross-hotel search.

---

**Related:** [[01 - Strategy Pattern]] | [[02 - Observer Pattern]]
