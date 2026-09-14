---
date: "2026-04-06"
type: lld-question
difficulty: medium
tags: [lld, interview-prep, bowling-alley]
---

# Design a Bowling Alley

## 1. Problem Statement
Design a bowling scoring system following standard rules: 10 frames, strikes, spares, and the 10th frame bonus.

## 2. Key Implementation (Python)

```python
class BowlingGame:
    def __init__(self):
        self.rolls = []

    def roll(self, pins: int):
        self.rolls.append(pins)

    def score(self) -> int:
        total = 0
        roll_idx = 0
        for frame in range(10):
            if roll_idx >= len(self.rolls):
                break
            if self._is_strike(roll_idx):
                total += 10 + self._strike_bonus(roll_idx)
                roll_idx += 1
            elif self._is_spare(roll_idx):
                total += 10 + self._spare_bonus(roll_idx)
                roll_idx += 2
            else:
                total += self.rolls[roll_idx] + self._safe_get(roll_idx + 1)
                roll_idx += 2
        return total

    def _is_strike(self, idx: int) -> bool:
        return self._safe_get(idx) == 10

    def _is_spare(self, idx: int) -> bool:
        return self._safe_get(idx) + self._safe_get(idx + 1) == 10

    def _strike_bonus(self, idx: int) -> int:
        return self._safe_get(idx + 1) + self._safe_get(idx + 2)

    def _spare_bonus(self, idx: int) -> int:
        return self._safe_get(idx + 2)

    def _safe_get(self, idx: int) -> int:
        return self.rolls[idx] if idx < len(self.rolls) else 0

class BowlingAlley:
    def __init__(self, num_lanes: int):
        self.lanes = {i: BowlingLane(i) for i in range(1, num_lanes + 1)}

class BowlingLane:
    def __init__(self, lane_id: int):
        self.lane_id = lane_id
        self.game: BowlingGame = None
        self.players = []

    def start_game(self, player_names):
        self.players = player_names
        self.game = BowlingGame()

if __name__ == "__main__":
    game = BowlingGame()
    # Perfect game: 12 strikes = 300
    for _ in range(12):
        game.roll(10)
    print(f"Perfect game score: {game.score()}")  # 300
```

## 3. Follow-ups: **Multiplayer?** Track per-player games | **League scoring?** Aggregate across multiple games | **Display?** Observer for scoreboard updates.

---
**Related:** [[01 - Strategy Pattern]] | [[02 - Observer Pattern]]
