---
date: "2026-04-06"
type: lld-question
difficulty: hard
status: active
tags: [lld, interview-prep, chess-tournament, observer-pattern, state-pattern]
---

# Design an Online Auction System

## 1. Problem Statement
Design an auction platform supporting item listing, bidding, auto-bidding, auction lifecycle, and winner determination.

## 2. Key Implementation (Python)

```python
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading

class AuctionStatus(Enum):
    UPCOMING = "UPCOMING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"

class Bid:
    def __init__(self, bidder_id: str, amount: float):
        self.bidder_id = bidder_id
        self.amount = amount
        self.timestamp = datetime.now()

class AuctionItem:
    def __init__(self, item_id: str, title: str, description: str,
                 starting_price: float, seller_id: str,
                 end_time: datetime):
        self.item_id = item_id
        self.title = title
        self.description = description
        self.starting_price = starting_price
        self.seller_id = seller_id
        self.end_time = end_time
        self.current_price = starting_price
        self.status = AuctionStatus.UPCOMING
        self.bids: List[Bid] = []
        self.winner: Optional[str] = None
        self._observers: List = []  # Notification callbacks

    def place_bid(self, bidder_id: str, amount: float) -> bool:
        if self.status != AuctionStatus.ACTIVE:
            print("❌ Auction not active")
            return False
        if bidder_id == self.seller_id:
            print("❌ Seller can't bid on own item")
            return False
        if amount <= self.current_price:
            print(f"❌ Bid must exceed current price ${self.current_price:.2f}")
            return False

        bid = Bid(bidder_id, amount)
        self.bids.append(bid)
        self.current_price = amount
        print(f"💰 New bid: ${amount:.2f} by {bidder_id} on '{self.title}'")
        self._notify_observers(bid)
        return True

    def start(self):
        self.status = AuctionStatus.ACTIVE

    def end(self):
        self.status = AuctionStatus.COMPLETED
        if self.bids:
            self.winner = self.bids[-1].bidder_id
            print(f"🏆 Winner: {self.winner} at ${self.current_price:.2f}")
        else:
            print(f"No bids on '{self.title}'")

    def add_observer(self, callback):
        self._observers.append(callback)

    def _notify_observers(self, bid: Bid):
        for cb in self._observers:
            cb(self, bid)

class AuctionService:
    def __init__(self):
        self.auctions: Dict[str, AuctionItem] = {}

    def create_auction(self, **kwargs) -> AuctionItem:
        item = AuctionItem(**kwargs)
        self.auctions[item.item_id] = item
        return item

    def get_active_auctions(self) -> List[AuctionItem]:
        return [a for a in self.auctions.values() if a.status == AuctionStatus.ACTIVE]

    def search(self, query: str) -> List[AuctionItem]:
        q = query.lower()
        return [a for a in self.auctions.values()
                if q in a.title.lower() or q in a.description.lower()]
```

## 3. Patterns: **Observer** (notify bidders on outbid) | **State** (auction lifecycle) | **Strategy** (auction types — English, Dutch, sealed)

## 4. Follow-ups
- **Auto-bidding?** User sets max bid, system auto-increments.
- **Reserve price?** Minimum price below which item won't sell.
- **Anti-sniping?** Extend auction if bid placed in last 5 minutes.

---

**Related:** [[02 - Observer Pattern]] | [[13 - State Pattern]] | [[01 - Strategy Pattern]]
