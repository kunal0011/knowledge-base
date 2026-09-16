---
date: "2026-04-06"
type: lld-question
difficulty: easy
status: active
tags: [lld, interview-prep, snake-ladder, game, state-pattern]
---

# Design a Snake and Ladder Game

## 1. Problem Statement
Design a Snake and Ladder board game for multiple players with configurable board, snakes, and ladders.

## 2. Key Implementation (Python)

```python
import random
from typing import List, Dict, Tuple

class Player:
    def __init__(self, name: str):
        self.name = name
        self.position = 0

class Board:
    def __init__(self, size: int = 100,
                 snakes: Dict[int, int] = None,
                 ladders: Dict[int, int] = None):
        self.size = size
        self.snakes = snakes or {16: 6, 47: 26, 49: 11, 56: 53, 62: 19, 87: 24, 93: 73, 95: 75, 98: 78}
        self.ladders = ladders or {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 36: 44, 51: 67, 71: 91, 80: 100}

    def get_final_position(self, position: int) -> Tuple[int, str]:
        if position in self.snakes:
            return self.snakes[position], "🐍 Snake!"
        if position in self.ladders:
            return self.ladders[position], "🪜 Ladder!"
        return position, ""

class Dice:
    def __init__(self, num_dice: int = 1, faces: int = 6):
        self.num_dice = num_dice
        self.faces = faces

    def roll(self) -> int:
        return sum(random.randint(1, self.faces) for _ in range(self.num_dice))

class SnakeLadderGame:
    def __init__(self, player_names: List[str], board: Board = None):
        self.board = board or Board()
        self.players = [Player(name) for name in player_names]
        self.dice = Dice()
        self.current_idx = 0
        self.winner = None

    def play_turn(self) -> str:
        player = self.players[self.current_idx]
        roll = self.dice.roll()
        new_pos = player.position + roll

        if new_pos > self.board.size:
            msg = f"{player.name} rolled {roll} — can't move (would exceed {self.board.size})"
        else:
            final_pos, event = self.board.get_final_position(new_pos)
            old_pos = player.position
            player.position = final_pos
            msg = f"{player.name} rolled {roll}: {old_pos} → {new_pos}"
            if event:
                msg += f" → {final_pos} {event}"

            if player.position == self.board.size:
                self.winner = player
                return f"🎉 {player.name} WINS!"

        self.current_idx = (self.current_idx + 1) % len(self.players)
        return msg

    def play(self):
        while not self.winner:
            print(self.play_turn())
        print(f"\nGame over! Winner: {self.winner.name}")

if __name__ == "__main__":
    game = SnakeLadderGame(["Alice", "Bob", "Charlie"])
    game.play()
```

### Java

```java
package com.lld.snakeladder;

import java.util.*;

class Player {
    private final String name;
    private int position = 0;
    public Player(String name) { this.name = name; }
    public String getName() { return name; }
    public int getPosition() { return position; }
    public void setPosition(int p) { this.position = p; }
}

public class SnakeAndLadderGame {
    private final Map<Integer, Integer> jumpMap = new HashMap<>(); // snakes & ladders
    private final Queue<Player> players = new LinkedList<>();
    private final Random random = new Random();
    private static final int WINNING_POSITION = 100;
    private boolean won = false;

    public void addSnake(int head, int tail) { jumpMap.put(head, tail); }
    public void addLadder(int start, int end) { jumpMap.put(start, end); }
    public void addPlayer(Player p) { players.offer(p); }

    public synchronized void playTurn() {
        if (won || players.isEmpty()) return;
        Player current = players.poll();
        int roll = random.nextInt(6) + 1;
        int nextPos = current.getPosition() + roll;

        if (nextPos <= WINNING_POSITION) {
            if (jumpMap.containsKey(nextPos)) {
                nextPos = jumpMap.get(nextPos);
            }
            current.setPosition(nextPos);
            System.out.printf("%s rolled %d, moved to %d%n", current.getName(), roll, nextPos);
            if (nextPos == WINNING_POSITION) {
                won = true;
                System.out.println("🎉 " + current.getName() + " won the game!");
                return;
            }
        }
        players.offer(current);
    }
}
```



---

## Thread Safety Considerations

| Concern | Solution |
|---|---|
| Turn race conditions | `playTurn` synchronized to ensure strictly alternating FIFO player queue turns |

## Extensibility & SOLID Principles

| Principle | Architectural Implementation |
|---|---|
| **S** — Single Responsibility | `Dice` encapsulates randomness; `Board` holds jump layout; `Game` enforces win rules |
| **O** — Open/Closed | Jump rules (portals, traps) extensible by decorating the jump mapping strategy |

---

## 3. Follow-ups
- **Multiple dice?** Configurable `Dice(num_dice=2)`.
- **Power-ups?** Strategy pattern for special squares.
- **Special rules?** Roll 6 → extra turn. Three 6s → go back to start.

---

**Related:** [[13 - State Pattern]] | [[01 - Strategy Pattern]]
