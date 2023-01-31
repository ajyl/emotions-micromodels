"""
Utility functions for cache.
"""

import os
import json


def load_cache(cache_filepath):
    print("Loading cache...")
    cache = {}
    if os.path.isfile(cache_filepath):
        with open(cache_filepath, "r") as file_p:
            cache = json.load(file_p)

    print("Finished loading cache.")
    return cache
