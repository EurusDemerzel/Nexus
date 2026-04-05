import psutil

class SplitDecision:
    def __init__(self):
        self.thresholds = {(30, 50): 2, (70, 80): 0}
        self.default_split = 1

    def get_system_load(self):
        return psutil.cpu_percent(interval=0.2), psutil.virtual_memory().percent

    def decide(self, force_level=None):
        if force_level is not None:
            return force_level

        cpu, mem = self.get_system_load()
        for (cpu_max, mem_max), split in self.thresholds.items():
            if cpu <= cpu_max and mem <= mem_max:
                return split
        return self.default_split

    # Backward-compatible alias for previous call sites.
    def decide_split_point(self, force_level=None):
        return self.decide(force_level=force_level)

split_decision = SplitDecision()