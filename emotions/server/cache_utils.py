"""
Utility functions for cache.
"""

import os
import json


def load_cache(cache_filepath):
    print("Loading cache...")
    cache = {}
    if not os.path.isfile(cache_filepath):
        return cache

    with open(cache_filepath, "r") as file_p:
        cache = json.load(file_p)
    return cache
