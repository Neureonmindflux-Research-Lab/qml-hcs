#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend Registry
----------------
Lightweight in-memory registry for discoverable quantum backends and their
factory constructors. Provides a unified mechanism to register, query,
instantiate, and list available backends through a global module-level API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from .types import Capabilities, QuantumBackend as QuantumBackendProtocol
from .backend import BackendConfig


@dataclass(frozen=True)
class Entry:
    """
    Represents a registry entry describing a backend constructor and its metadata.

    Parameters
    ----------
    name : str
        Human-readable backend name.
    constructor : Callable[[BackendConfig], QuantumBackendProtocol]
        Callable that returns an initialized backend instance when given
        a ``BackendConfig``.
    capabilities : Capabilities
        Static metadata describing backend capabilities.
    """

    name: str
    constructor: Callable[[BackendConfig], QuantumBackendProtocol]
    capabilities: Capabilities


class BackendRegistry:
    """
    In-memory registry that stores and manages backend entries.

    Methods allow registration, lookup, creation, and listing of all
    available backends.
    """

    def __init__(self) -> None:
        """Initialize an empty backend registry."""
        self._items: Dict[str, Entry] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        constructor: Callable[[BackendConfig], QuantumBackendProtocol],
        capabilities: Capabilities,
        overwrite: bool = False,
    ) -> None:
        """
        Register a new backend with the given name and constructor.

        Parameters
        ----------
        name : str
            Unique backend name.
        constructor : Callable[[BackendConfig], QuantumBackendProtocol]
            Constructor function returning a backend instance.
        capabilities : Capabilities
            Metadata describing backend features and limits.
        overwrite : bool, optional
            Whether to overwrite an existing entry with the same name.
            Default is ``False``.

        Raises
        ------
        ValueError
            If ``name`` is empty.
        KeyError
            If the backend name already exists and ``overwrite=False``.
        """
        key = name.strip().lower()
        if not key:
            raise ValueError("Backend name must be non-empty.")
        if key in self._items and not overwrite:
            raise KeyError(f"Backend '{name}' is already registered.")
        self._items[key] = Entry(
            name=name,
            constructor=constructor,
            capabilities=capabilities,
        )

    def exists(self, name: str) -> bool:
        """
        Check whether a backend with the given name exists.

        Parameters
        ----------
        name : str
            Backend name.

        Returns
        -------
        bool
            True if the backend is registered, False otherwise.
        """
        return name.strip().lower() in self._items

    def get(self, name: str) -> Entry:
        """
        Retrieve a backend entry by name.

        Parameters
        ----------
        name : str
            Backend name.

        Returns
        -------
        Entry
            Registry entry containing constructor and capabilities.

        Raises
        ------
        KeyError
            If the backend is not registered.
        """
        key = name.strip().lower()
        try:
            return self._items[key]
        except KeyError as e:
            raise KeyError(f"Backend '{name}' is not registered.") from e

    def create(self, name: str, config: BackendConfig) -> QuantumBackendProtocol:
        """
        Instantiate a backend by name.

        Parameters
        ----------
        name : str
            Backend name to instantiate.
        config : BackendConfig
            Configuration object to pass to the backend constructor.

        Returns
        -------
        QuantumBackendProtocol
            New backend instance.
        """
        entry = self.get(name)
        return entry.constructor(config)

    def list(self) -> Dict[str, Capabilities]:
        """
        List all registered backends and their capabilities.

        Returns
        -------
        dict[str, Capabilities]
            Dictionary mapping backend names to their capability metadata.
        """
        return {k: v.capabilities for k, v in self._items.items()}


# ----------------------------------------------------------------------
# Global singleton and public API
# ----------------------------------------------------------------------
_registry = BackendRegistry()


def register_backend(
    name: str,
    constructor: Callable[[BackendConfig], QuantumBackendProtocol],
    capabilities: Capabilities,
    overwrite: bool = False,
) -> None:
    """
    Register a backend globally.

    Parameters
    ----------
    name : str
        Backend name.
    constructor : Callable[[BackendConfig], QuantumBackendProtocol]
        Constructor function returning a backend instance.
    capabilities : Capabilities
        Backend capabilities metadata.
    overwrite : bool, optional
        Whether to overwrite an existing entry, by default ``False``.
    """
    _registry.register(name, constructor, capabilities, overwrite=overwrite)


def create_backend(name: str, config: BackendConfig) -> QuantumBackendProtocol:
    """
    Instantiate a registered backend by name.

    Parameters
    ----------
    name : str
        Backend name.
    config : BackendConfig
        Configuration for backend creation.

    Returns
    -------
    QuantumBackendProtocol
        Instantiated backend object.
    """
    return _registry.create(name, config)


def backend_exists(name: str) -> bool:
    """
    Check whether a backend is registered.

    Parameters
    ----------
    name : str
        Backend name.

    Returns
    -------
    bool
        True if the backend exists, False otherwise.
    """
    return _registry.exists(name)


def list_backends() -> Dict[str, Capabilities]:
    """
    List all registered backends.

    Returns
    -------
    dict[str, Capabilities]
        Mapping of backend names to their declared capabilities.
    """
    return _registry.list()
