---
date: "2026-04-06"
type: lld-question
difficulty: medium
status: active
tags: [lld, interview-prep, thread-pool, command-pattern]
---

# Design a Thread Pool

## 1. Problem Statement
Design a thread pool executor supporting task submission, fixed/dynamic thread count, and graceful shutdown.

## 2. Key Implementation (Python)

```python
import threading, queue, time
from typing import Callable, Optional
from enum import Enum

class ThreadPoolState(Enum):
    RUNNING = "RUNNING"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    TERMINATED = "TERMINATED"

class ThreadPool:
    def __init__(self, num_threads: int, queue_size: int = 100):
        self.num_threads = num_threads
        self._task_queue = queue.Queue(maxsize=queue_size)
        self._threads = []
        self._state = ThreadPoolState.RUNNING
        self._lock = threading.Lock()
        self._completed = 0

        for i in range(num_threads):
            t = threading.Thread(target=self._worker, name=f"Worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def submit(self, task: Callable, *args, **kwargs):
        if self._state != ThreadPoolState.RUNNING:
            raise RuntimeError("ThreadPool is shut down")
        self._task_queue.put((task, args, kwargs))

    def _worker(self):
        while True:
            try:
                task, args, kwargs = self._task_queue.get(timeout=1)
            except queue.Empty:
                if self._state == ThreadPoolState.SHUTTING_DOWN:
                    break
                continue
            try:
                task(*args, **kwargs)
            except Exception as e:
                print(f"Task error: {e}")
            finally:
                with self._lock:
                    self._completed += 1
                self._task_queue.task_done()

    def shutdown(self, wait: bool = True):
        self._state = ThreadPoolState.SHUTTING_DOWN
        if wait:
            self._task_queue.join()
        for t in self._threads:
            t.join(timeout=5)
        self._state = ThreadPoolState.TERMINATED
        print(f"ThreadPool terminated. {self._completed} tasks completed.")

    @property
    def pending_tasks(self) -> int:
        return self._task_queue.qsize()

    @property
    def completed_tasks(self) -> int:
        return self._completed

if __name__ == "__main__":
    pool = ThreadPool(num_threads=3)

    def task(n):
        print(f"  [{threading.current_thread().name}] Processing task {n}")
        time.sleep(0.2)

    for i in range(10):
        pool.submit(task, i)

    pool.shutdown(wait=True)
```

## 3. Java Implementation Sketch

```java
public class SimpleThreadPool {
    private final BlockingQueue<Runnable> taskQueue;
    private final List<Thread> workers;
    private volatile boolean isShutdown = false;

    public SimpleThreadPool(int numThreads, int queueSize) {
        taskQueue = new LinkedBlockingQueue<>(queueSize);
        workers = new ArrayList<>();
        for (int i = 0; i < numThreads; i++) {
            Thread t = new Thread(() -> {
                while (!isShutdown || !taskQueue.isEmpty()) {
                    try {
                        Runnable task = taskQueue.poll(1, TimeUnit.SECONDS);
                        if (task != null) task.run();
                    } catch (InterruptedException e) { break; }
                }
            });
            t.start();
            workers.add(t);
        }
    }

    public void submit(Runnable task) {
        if (isShutdown) throw new RejectedExecutionException();
        taskQueue.offer(task);
    }

    public void shutdown() { isShutdown = true; }
}
```

## 4. Rejection Policies (like Java's ThreadPoolExecutor)
| Policy | Behavior |
|--------|----------|
| **AbortPolicy** | Throw exception |
| **CallerRunsPolicy** | Execute in caller's thread |
| **DiscardPolicy** | Silently drop task |
| **DiscardOldestPolicy** | Drop oldest queued task |

## 5. Follow-ups
- **Dynamic sizing?** Min/max threads, scale based on queue depth.
- **Priority tasks?** Use `PriorityQueue` instead of FIFO.
- **Future/Promise?** Return `Future` from submit for async result.

---

**Related:** [[07 - Command Pattern]] | [[01 - Strategy Pattern]]
