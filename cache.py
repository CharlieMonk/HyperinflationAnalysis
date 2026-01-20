"""Caching utilities for hyperinflation analysis."""

import os
import pickle

from config import DEFAULT_CACHE_DIR


def _date_to_cache_str(dt):
    """Convert datetime to string for cache key."""
    if hasattr(dt, 'strftime'):
        return dt.strftime('%Y%m')
    return str(dt)[:7].replace('-', '')


def load_or_download(cache_file, download_func, description, cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """Load from cache if exists, otherwise download and cache.

    Args:
        cache_file: Name of the cache file
        download_func: Callable that downloads and returns the data
        description: Description for verbose output
        cache_dir: Directory for cache files
        verbose: Print progress messages

    Returns:
        Cached or freshly downloaded data
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    else:
        if verbose:
            print(f"  Downloading {description}...")
        data = download_func()
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        return data
