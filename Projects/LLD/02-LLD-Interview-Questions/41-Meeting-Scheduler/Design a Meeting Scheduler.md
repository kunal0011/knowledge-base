---
date: "2026-04-06"
type: lld-question
difficulty: medium
tags: [lld, interview-prep, meeting-scheduler]
---

# Design a Meeting Scheduler

## 1. Problem Statement
Design a calendar meeting scheduler supporting room booking, conflict detection, and recurring meetings.

## 2. Key Implementation (Python)

```python
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import uuid

class TimeSlot:
    def __init__(self, start: datetime, end: datetime):
        self.start = start; self.end = end
    def overlaps(self, other: 'TimeSlot') -> bool:
        return self.start < other.end and other.start < self.end

class MeetingRoom:
    def __init__(self, room_id: str, name: str, capacity: int):
        self.room_id = room_id; self.name = name; self.capacity = capacity
        self._bookings: List['Meeting'] = []

    def is_available(self, slot: TimeSlot) -> bool:
        return not any(m.time_slot.overlaps(slot) for m in self._bookings)

    def book(self, meeting: 'Meeting') -> bool:
        if self.is_available(meeting.time_slot):
            self._bookings.append(meeting); return True
        return False

class Meeting:
    def __init__(self, title: str, organizer: str, time_slot: TimeSlot,
                 participants: List[str], room: Optional[MeetingRoom] = None):
        self.meeting_id = str(uuid.uuid4())[:8]
        self.title = title; self.organizer = organizer
        self.time_slot = time_slot; self.participants = participants
        self.room = room

class CalendarService:
    def __init__(self):
        self.rooms: Dict[str, MeetingRoom] = {}
        self.user_meetings: Dict[str, List[Meeting]] = {}

    def add_room(self, room: MeetingRoom):
        self.rooms[room.room_id] = room

    def schedule_meeting(self, title: str, organizer: str, slot: TimeSlot,
                         participants: List[str], capacity: int) -> Optional[Meeting]:
        # Check participant conflicts
        for p in participants + [organizer]:
            for m in self.user_meetings.get(p, []):
                if m.time_slot.overlaps(slot):
                    print(f"❌ Conflict: {p} busy during {m.title}")
                    return None

        # Find available room
        room = next((r for r in self.rooms.values()
                     if r.capacity >= capacity and r.is_available(slot)), None)
        if not room:
            print("❌ No room available"); return None

        meeting = Meeting(title, organizer, slot, participants, room)
        room.book(meeting)
        for p in participants + [organizer]:
            self.user_meetings.setdefault(p, []).append(meeting)
        print(f"✅ Scheduled: {title} in {room.name}")
        return meeting

    def find_available_slots(self, participants: List[str], duration_min: int,
                             date: datetime) -> List[TimeSlot]:
        """Find free slots for all participants on a given date"""
        slots = []
        start = date.replace(hour=9, minute=0)
        end_of_day = date.replace(hour=18, minute=0)
        while start + timedelta(minutes=duration_min) <= end_of_day:
            slot = TimeSlot(start, start + timedelta(minutes=duration_min))
            free = all(not any(m.time_slot.overlaps(slot) for m in self.user_meetings.get(p, []))
                      for p in participants)
            if free:
                slots.append(slot)
            start += timedelta(minutes=30)
        return slots
```

## 3. Patterns: **Strategy** (room selection) | **Observer** (calendar notifications) | **Composite** (recurring meetings)
## 4. Follow-ups: **Recurring?** Template generates instances | **Timezone?** Store in UTC, display in user's TZ | **Integration?** Adapter for Google Calendar API.

---
**Related:** [[01 - Strategy Pattern]] | [[02 - Observer Pattern]]
