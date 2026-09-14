---
date: "2026-04-06"
type: lld-question
difficulty: easy
status: active
tags: [lld, interview-prep, deck-of-cards]
---

# Design a Deck of Cards

## 1. Problem Statement
Design an OOP model for a standard deck of cards supporting shuffling, dealing, and games like Blackjack.

## 2. Key Implementation (Python)

```python
from enum import Enum
from typing import List, Optional
import random

class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"

class Rank(Enum):
    ACE = (1, "A")
    TWO = (2, "2"); THREE = (3, "3"); FOUR = (4, "4"); FIVE = (5, "5")
    SIX = (6, "6"); SEVEN = (7, "7"); EIGHT = (8, "8"); NINE = (9, "9")
    TEN = (10, "10"); JACK = (10, "J"); QUEEN = (10, "Q"); KING = (10, "K")

class Card:
    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank
    def __str__(self):
        return f"{self.rank.value[1]}{self.suit.value}"

class Deck:
    def __init__(self):
        self.cards: List[Card] = [Card(s, r) for s in Suit for r in Rank]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self) -> Optional[Card]:
        return self.cards.pop() if self.cards else None

    def deal_hand(self, count: int) -> List[Card]:
        return [self.deal() for _ in range(min(count, len(self.cards)))]

    def remaining(self) -> int:
        return len(self.cards)

class Hand:
    def __init__(self):
        self.cards: List[Card] = []

    def add_card(self, card: Card):
        self.cards.append(card)

    def score(self) -> int:
        """Blackjack scoring — Aces are 1 or 11"""
        total = sum(c.rank.value[0] for c in self.cards)
        aces = sum(1 for c in self.cards if c.rank == Rank.ACE)
        while total <= 11 and aces > 0:
            total += 10; aces -= 1
        return total

    def __str__(self):
        return ' '.join(str(c) for c in self.cards) + f" (score: {self.score()})"

class BlackjackGame:
    def __init__(self):
        self.deck = Deck()
        self.deck.shuffle()
        self.player = Hand()
        self.dealer = Hand()

    def deal_initial(self):
        for _ in range(2):
            self.player.add_card(self.deck.deal())
            self.dealer.add_card(self.deck.deal())

    def hit(self, hand: Hand):
        hand.add_card(self.deck.deal())

    def is_bust(self, hand: Hand) -> bool:
        return hand.score() > 21
```

## 3. Follow-ups: **Multiple games?** Strategy pattern per game type | **Multiplayer?** Player class with hand management | **Betting?** Observer for bet resolution.

---

**Related:** [[01 - Strategy Pattern]] | [[10 - Template Method Pattern]]
