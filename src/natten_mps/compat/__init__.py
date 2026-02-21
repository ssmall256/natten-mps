"""
Compatibility shims for upstream NATTEN API versions.
"""

from __future__ import annotations


def for_version(version_string: str):
    """Return the appropriate compat module for a given natten version string."""
    from packaging.version import Version

    v = Version(version_string)
    if v < Version("0.15.0"):
        from . import v014

        return v014
    if v < Version("0.20.0"):
        from . import v017

        return v017
    from . import v020

    return v020


__all__ = ["for_version"]
