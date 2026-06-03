"""Custom exceptions for clearer command-line failures."""

from __future__ import annotations


class WaveSleuthError(Exception):
    """Base exception for expected WaveSleuth failures."""


class ValidationError(WaveSleuthError):
    """Raised when a world, run, or reconstruction file is malformed."""


class UnsupportedWorldError(WaveSleuthError):
    """Raised when an MVP feature does not support the requested world type."""


class DevitoUnavailableError(WaveSleuthError):
    """Raised when a simulation command needs Devito but it is unavailable."""
