---
date: "2026-04-06"
type: lld-question
difficulty: medium
status: active
tags: [lld, interview-prep, airline-booking, state-pattern]
---

# Design an Airline Booking System

## 1. Problem Statement
Design an airline reservation system with flight search, seat selection, booking, and cancellation.

## 2. Key Implementation (Python)

```python
from enum import Enum
from datetime import datetime
from typing import List, Dict, Optional
import uuid

class SeatClass(Enum):
    ECONOMY = ("Economy", 100)
    BUSINESS = ("Business", 300)
    FIRST = ("First", 500)

class SeatStatus(Enum):
    AVAILABLE = "AVAILABLE"
    BOOKED = "BOOKED"
    BLOCKED = "BLOCKED"

class Seat:
    def __init__(self, seat_id: str, seat_class: SeatClass):
        self.seat_id = seat_id
        self.seat_class = seat_class
        self.status = SeatStatus.AVAILABLE

class Flight:
    def __init__(self, flight_no: str, origin: str, dest: str,
                 departure: datetime, arrival: datetime):
        self.flight_no = flight_no
        self.origin = origin
        self.destination = dest
        self.departure = departure
        self.arrival = arrival
        self.seats: Dict[str, Seat] = {}

    def add_seats(self, seat_class: SeatClass, count: int):
        prefix = seat_class.name[0]
        for i in range(count):
            sid = f"{prefix}{i+1}"
            self.seats[sid] = Seat(sid, seat_class)

    def get_available(self, seat_class: Optional[SeatClass] = None) -> List[Seat]:
        return [s for s in self.seats.values()
                if s.status == SeatStatus.AVAILABLE and
                (seat_class is None or s.seat_class == seat_class)]

class Booking:
    def __init__(self, passenger_name: str, flight: Flight, seat: Seat):
        self.booking_id = str(uuid.uuid4())[:8]
        self.passenger_name = passenger_name
        self.flight = flight
        self.seat = seat
        self.fare = seat.seat_class.value[1]
        self.status = "CONFIRMED"

    def cancel(self):
        self.seat.status = SeatStatus.AVAILABLE
        self.status = "CANCELLED"
        print(f"❌ Booking {self.booking_id} cancelled. Seat {self.seat.seat_id} released.")

class AirlineSystem:
    def __init__(self):
        self.flights: Dict[str, Flight] = {}
        self.bookings: Dict[str, Booking] = {}

    def search_flights(self, origin: str, dest: str, date: datetime) -> List[Flight]:
        return [f for f in self.flights.values()
                if f.origin == origin and f.destination == dest
                and f.departure.date() == date.date()]

    def book_flight(self, flight_no: str, seat_id: str, passenger: str) -> Optional[Booking]:
        flight = self.flights.get(flight_no)
        if not flight:
            return None
        seat = flight.seats.get(seat_id)
        if not seat or seat.status != SeatStatus.AVAILABLE:
            print("❌ Seat not available")
            return None
        seat.status = SeatStatus.BOOKED
        booking = Booking(passenger, flight, seat)
        self.bookings[booking.booking_id] = booking
        print(f"✅ Booked: {passenger} on {flight_no}, seat {seat_id} (${booking.fare})")
        return booking
```

## 3. Patterns: **State** (booking lifecycle) | **Strategy** (pricing models) | **Observer** (flight status updates)

## 4. Follow-ups
- **Waitlist?** Queue per flight. Observer notifies when seat freed.
- **Dynamic pricing?** Strategy — demand-based fare adjustment.
- **Multi-leg journeys?** Composite pattern for connecting flights.

---

**Related:** [[13 - State Pattern]] | [[01 - Strategy Pattern]]
