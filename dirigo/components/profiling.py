import contextvars, time, math
from dataclasses import dataclass
from queue import SimpleQueue, Empty
from threading import Thread, Event


run_id_var     = contextvars.ContextVar("dirigo_run_id", default=None)
group_var      = contextvars.ContextVar("dirigo_group",  default=None)
plugin_var     = contextvars.ContextVar("dirigo_plugin", default=None)
worker_var     = contextvars.ContextVar("dirigo_worker", default=None)  # optional human name


@dataclass(frozen=True)
class MetricEvent:
    run_id: str | None
    group:  str | None       # "acquisition" | "processor" | "writer" | ...
    plugin: str | None       # entry point name, e.g., "raster_line"
    worker: str | None       # thread name or friendly name
    stage:  str              # e.g., "digitizer.get_completed"
    wall_ns: int
    cpu_ns:  int
    n:       int = 1


_metrics_queue = SimpleQueue()


class _Timer:
    __slots__ = ("stage", "t0w", "t0c")
    def __init__(self, stage: str): 
        self.stage = stage

    def __enter__(self):
        self.t0w = time.perf_counter_ns() # should these just be float seconds or is there a reason to do ns?
        self.t0c = time.thread_time()
        return self
    
    def __exit__(self, *_):
        wall = time.perf_counter_ns() - self.t0w
        cpu  = int((time.thread_time() - self.t0c) * 1e9)
        _metrics_queue.put(MetricEvent(
            run_id_var.get(), 
            group_var.get(), 
            plugin_var.get(), 
            worker_var.get(),
            self.stage, 
            wall, 
            cpu
        ))


def timer(stage: str):       # Worker code just does: with timer("publish"):
    return _Timer(stage)


def incr(stage: str, n: int = 1):
    _metrics_queue.put(MetricEvent(
        run_id_var.get(), 
        group_var.get(), 
        plugin_var.get(), 
        worker_var.get(),
        stage, 
        0, 
        0, 
        n)
    )


class MetricsAggregator(Thread): # Should this be a dirigo.Worker?
    def __init__(self, interval_s: float = 2.0):
        super().__init__(daemon=True)
        self.stop_ev = Event()
        self.interval_s = interval_s
        self.counters = {}    # ((worker,stage),"count")->int
        self.wall = {}        # (worker,stage)->{bucket:int}
        self.cpu  = {}        # (worker,stage)->{bucket:int}

    @staticmethod
    def _bucket(ns: int) -> int:
        return 0 if ns <= 0 else int(math.log2(ns))

    def _bump(self, dct, key, bucket):
        m = dct.setdefault(key, {})
        m[bucket] = m.get(bucket, 0) + 1

    def run(self):
        last = time.time()
        while not self.stop_ev.is_set():
            try:
                ev = _metrics_queue.get(timeout=0.1)
                key = (ev.worker, ev.stage)
                self.counters[(key,"count")] = self.counters.get((key,"count"),0) + ev.n
                if ev.wall_ns: self._bump(self.wall, key, self._bucket(ev.wall_ns))
                if ev.cpu_ns:  self._bump(self.cpu,  key, self._bucket(ev.cpu_ns))
            except Empty:
                pass
            if time.time() - last >= self.interval_s:
                self._emit_snapshot()
                last = time.time()

    def _emit_snapshot(self):
        lines = []
        keys = {k for (k, tag) in self.counters.keys() if tag == "count"}
        for key in sorted(keys):
            cnt = self.counters.get((key,"count"), 0)
            p50w, p95w, p99w = self._percentiles(self.wall.get(key, {}))
            p50c, p95c, p99c = self._percentiles(self.cpu.get(key,  {}))
            worker, stage = key
            lines.append(f"{worker}.{stage}: n={cnt} "
                         f"wall p50/p95/p99={p50w}/{p95w}/{p99w} µs "
                         f"cpu p50/p95/p99={p50c}/{p95c}/{p99c} µs")
        if lines:
            print("[metrics]", *lines, sep="\n")

    @staticmethod
    def _percentiles(hist):
        if not hist: return (0,0,0)
        total = sum(hist.values()); acc = 0
        targets = [0.5*total, 0.95*total, 0.99*total]
        out = []
        for b in sorted(hist):
            acc += hist[b]
            while targets and acc >= targets[0]:
                ns = (2**b + 2**(b+1))//2
                out.append(int(ns/1000))  # µs
                targets.pop(0)
        while len(out)<3: out.append(out[-1] if out else 0)
        return tuple(out)