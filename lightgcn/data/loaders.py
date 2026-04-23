"""
Data Loading Utilities

Functions for loading interaction data from files.
"""

from typing import Dict, List, Tuple, Any, Optional, Union


def _parse_timestamp(value: str) -> Optional[int]:
    value = value.strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return None


def load_interactions_from_file(
    filepath: str,
    separator: str = None,  # Auto-detect if None
    has_header: bool = None,  # Auto-detect if None
    user_col: int = 0,
    item_col: int = 1,
    timestamp_col: Optional[int] = None,
    max_rows: int = None
) -> List[Union[Tuple[Any, Any], Tuple[Any, Any, int]]]:
    """
    Load user-item interactions from a file.
    
    Supports both numeric and string IDs (will be reindexed later).
    Auto-detects CSV vs TSV format and header presence.
    
    Args:
        filepath: Path to interaction file
        separator: Column separator (auto-detect if None)
        has_header: Whether file has header row (auto-detect if None)
        user_col: Column index for user ID
        item_col: Column index for item ID
        max_rows: Maximum rows to load (None for all)
    
    Returns:
        List of (user_id, item_id) tuples
    """
    
    # Read first line to detect format
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    
    # Auto-detect separator: use tabs for .inter files or if more tabs than commas
    if separator is None:
        if filepath.endswith('.inter') or first_line.count('\t') > first_line.count(','):
            separator = '\t'
        else:
            separator = ','
    
    parts = first_line.split(separator)
    
    # Auto-detect header: if first column looks like 'user_id', 'user', etc.
    if has_header is None:
        has_header = any(
            keyword in parts[0].lower() 
            for keyword in ['user', 'id', 'asin', 'item', 'rating']
        )
    
    if has_header and timestamp_col is None:
        for idx, name in enumerate(parts):
            key = name.strip().lower()
            if key in {'timestamp', 'time', 'ts', 'unix', 'datetime'}:
                timestamp_col = idx
                break

    interactions: List[Union[Tuple[Any, Any], Tuple[Any, Any, int]]] = []
    
    with open(filepath, 'r') as f:
        if has_header:
            next(f)
        
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            
            parts = line.strip().split(separator)
            if len(parts) > max(user_col, item_col):
                user = parts[user_col].strip()
                item = parts[item_col].strip()
                if timestamp_col is not None and len(parts) > timestamp_col:
                    ts = _parse_timestamp(parts[timestamp_col])
                    if ts is None:
                        continue
                    interactions.append((user, item, ts))
                else:
                    interactions.append((user, item))
            
            # Progress logging for large files
            if (i + 1) % 1000000 == 0:
                print(f"  Loaded {(i + 1) // 1000000}M interactions...")
    
    return interactions


def reindex_interactions(
    interactions: List[Tuple[Any, ...]]
) -> Tuple[List[Tuple[int, ...]], Dict[Any, int], Dict[Any, int]]:
    """
    Reindex user and item IDs to be consecutive integers starting from 0.
    
    Args:
        interactions: List of (user_id, item_id) tuples, optionally with timestamps
    
    Returns:
        reindexed_interactions: List with new integer IDs (timestamp preserved if present)
        user_mapping: {original_id: new_id}
        item_mapping: {original_id: new_id}
    """
    users = set()
    items = set()
    
    for row in interactions:
        if len(row) < 2:
            continue
        u, i = row[0], row[1]
        users.add(u)
        items.add(i)
    
    user_mapping = {u: idx for idx, u in enumerate(sorted(users))}
    item_mapping = {i: idx for idx, i in enumerate(sorted(items))}
    
    reindexed = []
    for row in interactions:
        if len(row) < 2:
            continue
        u, i = row[0], row[1]
        if len(row) >= 3:
            reindexed.append((user_mapping[u], item_mapping[i], row[2]))
        else:
            reindexed.append((user_mapping[u], item_mapping[i]))
    
    return reindexed, user_mapping, item_mapping

