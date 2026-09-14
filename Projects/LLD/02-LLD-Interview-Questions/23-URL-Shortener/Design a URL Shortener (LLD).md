---
date: "2026-04-06"
type: lld-question
difficulty: medium
status: active
tags: [lld, interview-prep, url-shortener]
---

# Design a URL Shortener (LLD)

## 1. Problem Statement
Design the LLD for a URL shortener like bit.ly — creating short URLs, redirecting, tracking analytics.

## 2. Key Implementation (Python)

```python
import hashlib, string, time
from typing import Dict, Optional
from datetime import datetime

class URLShortener:
    BASE62 = string.ascii_letters + string.digits  # 62 chars
    
    def __init__(self, domain: str = "short.ly"):
        self.domain = domain
        self._url_map: Dict[str, str] = {}      # short → long
        self._reverse_map: Dict[str, str] = {}  # long → short
        self._analytics: Dict[str, list] = {}   # short → [click timestamps]
        self._counter = 100000  # Auto-increment counter

    def shorten(self, long_url: str, custom_alias: str = None) -> str:
        # Check if already shortened
        if long_url in self._reverse_map:
            return f"https://{self.domain}/{self._reverse_map[long_url]}"
        
        if custom_alias:
            if custom_alias in self._url_map:
                raise ValueError(f"Alias '{custom_alias}' already taken")
            short_code = custom_alias
        else:
            short_code = self._generate_code()

        self._url_map[short_code] = long_url
        self._reverse_map[long_url] = short_code
        self._analytics[short_code] = []
        return f"https://{self.domain}/{short_code}"

    def redirect(self, short_code: str) -> Optional[str]:
        long_url = self._url_map.get(short_code)
        if long_url:
            self._analytics[short_code].append(datetime.now())
        return long_url

    def get_analytics(self, short_code: str) -> dict:
        clicks = self._analytics.get(short_code, [])
        return {"total_clicks": len(clicks),
                "last_click": clicks[-1] if clicks else None}

    def _generate_code(self) -> str:
        """Base62 encode an auto-incrementing counter"""
        self._counter += 1
        return self._base62_encode(self._counter)

    def _base62_encode(self, num: int) -> str:
        if num == 0:
            return self.BASE62[0]
        result = []
        while num:
            result.append(self.BASE62[num % 62])
            num //= 62
        return ''.join(reversed(result))
```

## 3. Short Code Generation Strategies
| Strategy | Approach | Pros | Cons |
|----------|----------|------|------|
| **Base62 Counter** | Auto-increment → base62 | No collision, predictable | Sequential, guessable |
| **MD5/SHA Hash** | Hash URL → take first 7 chars | Content-based | Collision possible |
| **Random** | Random 7-char string | Non-guessable | Collision check needed |
| **Snowflake ID** | Distributed unique IDs | Globally unique | Complex setup |

## 4. Follow-ups
- **Expiration?** TTL per URL, background cleanup job.
- **Rate limiting?** Per-user creation limits.
- **Custom domains?** Multi-tenant with domain mapping.

---

**Related:** [[01 - Strategy Pattern]] | [[06 - Singleton Pattern]]
