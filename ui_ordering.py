"""Pure helpers for keeping free-angle upload order stable."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path


def upload_path(upload) -> str:
    """Return a filesystem path for Gradio upload values and test doubles."""
    if isinstance(upload, (str, bytes, Path)) or hasattr(upload, "__fspath__"):
        return os.fspath(upload)
    for attribute in ("path", "name"):
        value = getattr(upload, attribute, None)
        if value:
            return os.fspath(value)
    return os.fspath(upload)


def normalize_uploads(uploads) -> list[str]:
    return [upload_path(upload) for upload in list(uploads or []) if upload is not None]


def sync_upload_order(current_order, known_uploads, uploads) -> tuple[list[str], list[str]]:
    """Merge uploaded/removed files without discarding a custom card order."""
    current = normalize_uploads(current_order)
    known = normalize_uploads(known_uploads)
    uploaded = normalize_uploads(uploads)

    if not current:
        return uploaded, uploaded

    same_files = Counter(known) == Counter(uploaded)
    if same_files and uploaded != known:
        return uploaded, uploaded
    if same_files:
        return current, uploaded

    remaining = Counter(uploaded)
    merged: list[str] = []
    for path in current:
        if remaining[path] > 0:
            merged.append(path)
            remaining[path] -= 1
    for path in uploaded:
        if remaining[path] > 0:
            merged.append(path)
            remaining[path] -= 1
    return merged, uploaded


def apply_index_order(order, indices) -> list[str]:
    """Apply a browser card permutation after validating it exactly."""
    paths = normalize_uploads(order)
    parsed = [int(index) for index in list(indices or [])]
    if sorted(parsed) != list(range(len(paths))):
        raise ValueError("The browser returned an invalid image order. Please try again.")
    return [paths[index] for index in parsed]


def selection_choices(order) -> list[tuple[str, str]]:
    return [
        (f"{index + 1} - {Path(path).name}", str(index))
        for index, path in enumerate(normalize_uploads(order))
    ]


def move_selected(order, selected, action: str) -> tuple[list[str], int, str]:
    """Move a selected card and return its new index and a short status."""
    paths = normalize_uploads(order)
    if not paths:
        raise ValueError("Upload at least one free-angle image first.")
    if selected in (None, ""):
        raise ValueError("Choose a thumbnail from the selection list first.")
    index = int(selected)
    if not 0 <= index < len(paths):
        raise ValueError("The selected thumbnail is no longer available.")

    if action == "first":
        target = 0
    elif action == "earlier":
        target = max(0, index - 1)
    elif action == "later":
        target = min(len(paths) - 1, index + 1)
    else:
        raise ValueError(f"Unknown reorder action: {action}")

    item = paths.pop(index)
    paths.insert(target, item)
    return paths, target, f"Moved {Path(item).name} to position {target + 1}."
