# Copyright (c) Facebook, Inc. and its affiliates.

import time


class TimeChecker:
    """Utility class that provides playback time related functionality.
    TimeChecker starts running the clock when it is initialized. `get_time()`
    method can be used to query current playback time.

    Attributes:
        start: Stores unix time at the start of visualization
        data: Dictionary to store messages with timestamps. Use `save()` to
            record messages and `get_data()` or `print_data()` to retrieve
            messages
    """
    def __init__(self):
        self.start = 0.0
        self.data = []
        self.begin()

    def begin(self):
        self.start = time.time()
        del self.data[:]

    def print_time(self, restart=True):
        print(f"Time elapsed: {self.get_time(restart)}")

    def get_time(self, restart=True):
        t = time.time() - self.start
        if restart:
            self.begin()
        return t

    def print_data(self):
        print(self.data)

    def get_data(self):
        return self.data

    def save(self, msg=" "):
        self.data.append([self.get_time(), msg])
