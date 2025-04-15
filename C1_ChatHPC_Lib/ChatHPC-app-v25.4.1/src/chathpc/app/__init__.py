"""ChatHPC Application: A Python package for interacting with Kokkos-based applications.

This module provides the main components for creating and configuring
Kokkos-based applications using a chat-like interface.
"""

from chathpc.app.__about__ import __version__
from chathpc.app.app import App, AppConfig

version = __version__

__all__ = ["App", "AppConfig", "version"]
