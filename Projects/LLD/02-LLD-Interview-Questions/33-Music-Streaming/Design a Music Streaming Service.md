---
date: "2026-04-06"
type: lld-question
difficulty: medium
status: active
tags: [lld, interview-prep, music-player, state-pattern, observer-pattern]
---

# Design a Music Streaming Service (Spotify)

## 1. Problem Statement
Design a music streaming service with playlists, search, play queue, and user subscriptions.

## 2. Key Implementation (Python)

```python
from enum import Enum
from typing import List, Dict, Optional
from collections import deque

class Song:
    def __init__(self, song_id: str, title: str, artist: str,
                 album: str, duration_sec: int):
        self.song_id = song_id
        self.title = title
        self.artist = artist
        self.album = album
        self.duration = duration_sec

class Playlist:
    def __init__(self, name: str, owner_id: str):
        self.name = name
        self.owner_id = owner_id
        self.songs: List[Song] = []

    def add_song(self, song: Song):
        self.songs.append(song)

    def remove_song(self, song_id: str):
        self.songs = [s for s in self.songs if s.song_id != song_id]

class PlaybackState(Enum):
    STOPPED = "STOPPED"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"

class MusicPlayer:
    def __init__(self):
        self.state = PlaybackState.STOPPED
        self.current_song: Optional[Song] = None
        self.queue: deque = deque()
        self.history: List[Song] = []

    def play(self, song: Song):
        self.current_song = song
        self.state = PlaybackState.PLAYING
        self.history.append(song)
        print(f"▶️ Playing: {song.title} — {song.artist}")

    def pause(self):
        if self.state == PlaybackState.PLAYING:
            self.state = PlaybackState.PAUSED
            print(f"⏸️ Paused: {self.current_song.title}")

    def resume(self):
        if self.state == PlaybackState.PAUSED:
            self.state = PlaybackState.PLAYING
            print(f"▶️ Resumed: {self.current_song.title}")

    def next(self):
        if self.queue:
            song = self.queue.popleft()
            self.play(song)
        else:
            self.state = PlaybackState.STOPPED
            print("⏹️ Queue empty")

    def add_to_queue(self, song: Song):
        self.queue.append(song)
        print(f"➕ Added to queue: {song.title}")

    def play_playlist(self, playlist: Playlist, shuffle: bool = False):
        songs = list(playlist.songs)
        if shuffle:
            import random
            random.shuffle(songs)
        self.queue = deque(songs[1:])
        if songs:
            self.play(songs[0])

class MusicService:
    def __init__(self):
        self.songs: Dict[str, Song] = {}
        self.playlists: Dict[str, Playlist] = {}
        self.players: Dict[str, MusicPlayer] = {}

    def search(self, query: str) -> List[Song]:
        q = query.lower()
        return [s for s in self.songs.values()
                if q in s.title.lower() or q in s.artist.lower()]

    def get_player(self, user_id: str) -> MusicPlayer:
        if user_id not in self.players:
            self.players[user_id] = MusicPlayer()
        return self.players[user_id]
```

## 3. Patterns: **State** (playback state machine) | **Observer** (playback events) | **Strategy** (shuffle/repeat modes) | **Iterator** (queue traversal)

## 4. Follow-ups
- **Offline mode?** Download cache with DRM wrapper (Proxy pattern).
- **Recommendations?** Strategy for recommendation algorithms.
- **Lyrics sync?** Time-stamped lyrics mapped to playback position.
- **Collaborative playlists?** Observer — notify all collaborators on changes.

---

**Related:** [[13 - State Pattern]] | [[02 - Observer Pattern]] | [[11 - Iterator Pattern]]
