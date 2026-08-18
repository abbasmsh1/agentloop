"""Run event storage. The engine appends; frontends load."""

import json
import os


class JsonlRunStore:
    """One run, one .jsonl file. The default store."""

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)

    def append(self, entry):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            return []


class MemoryRunStore:
    """In-process store for tests and embedding."""

    name = "memory"

    def __init__(self):
        self.events = []

    def append(self, entry):
        self.events.append(dict(entry))

    def load(self):
        return list(self.events)
