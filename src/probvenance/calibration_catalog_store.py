"""Exact content-addressed directory store for calibration profile catalogs.

This module implements the first supported persistent store for exact catalog
snapshots. It answers exactly one question: given an exact catalog snapshot
fingerprint and fingerprint schema version, retrieve that exact catalog
snapshot.

It deliberately answers no lifecycle question. There is no latest catalog, no
active or default catalog, no production or staging channel, no alias, no
supersession, no refresh, and no enumeration. A directory containing catalog
snapshots is not a deployment registry: the caller must already know the exact
snapshot identity it wants.

Catalog storage is the outer layer of the architectural chain, and storing a
snapshot collapses none of the layers below it. A retrieved catalog still only
proposes candidate identities: exact profile retrieval from the profile store
and Phase 4C.7 selection remain mandatory before any profile can be applied.

Dependency direction: this module depends on
:mod:`probvenance.calibration_catalog` for the serialization contract and on the
shared filesystem helper and error foundations. The calibration, runtime,
results, trace, catalog, profile store, and selection modules never import this
one.

The store is content addressed. A snapshot's path is derived only from the
catalog store layout version, the catalog fingerprint schema version, and the
exact catalog fingerprint, and the stored bytes are exactly the canonical
serialization emitted by
:func:`probvenance.calibration_catalog.serialize_calibration_profile_catalog`.
Retrieval supplies the requested identity to the catalog loader as an
independent expected identity pin, so a different but internally
self-consistent catalog snapshot placed at the requested path is rejected.
"""

from __future__ import annotations

import os
from pathlib import Path

from probvenance._directory_artifact_store import (
    create_managed_directories,
    managed_path_exists,
    publish_atomically,
    read_managed_text,
    reject_non_directory_components,
    require_directory_root,
)
from probvenance._strict_json import abbreviate_untrusted
from probvenance.calibration_catalog import (
    CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION,
    CalibrationProfileCatalog,
    _require_catalog_fingerprint_identity,
    load_calibration_profile_catalog,
    serialize_calibration_profile_catalog,
)
from probvenance.errors import (
    CalibrationProfileCatalogNotFoundError,
    CalibrationProfileCatalogStoreError,
    CalibrationProfileCatalogStoreIntegrityError,
    InvalidDecisionError,
)

CALIBRATION_PROFILE_CATALOG_DIRECTORY_STORE_VERSION = 1
"""Version of the on-disk catalog store directory layout.

This is independent of the catalog fingerprint schema version, the catalog
serialization version, the profile fingerprint schema version, the profile
serialization version, and the profile store layout version. None of these
versions are interchangeable.
"""

_STORE_DIRECTORY = f"catalog-store-v{CALIBRATION_PROFILE_CATALOG_DIRECTORY_STORE_VERSION}"

#: Noun phrases the shared filesystem helper uses to phrase this store's own
#: messages. The helper is domain neutral, so the store supplies its own nouns.
_ARTIFACT_NAME = "calibration profile catalog"
_STORE_NAME = "calibration profile catalog store"


def _operational_error(detail: str) -> Exception:
    """Build this store's operational error for the shared helper."""
    return CalibrationProfileCatalogStoreError(detail)


def _integrity_error(detail: str) -> Exception:
    """Build this store's integrity error for the shared helper."""
    return CalibrationProfileCatalogStoreIntegrityError(detail)


def _managed_components(target: Path) -> tuple[tuple[Path, str], ...]:
    """Return the managed layout components that must be directories."""
    return (
        (target.parent.parent, "catalog store layout version directory"),
        (target.parent, "catalog fingerprint version directory"),
    )


class DirectoryCalibrationProfileCatalogStore:
    """Exact, content-addressed, directory-backed catalog snapshot store.

    The store is immutable in the sense that a catalog snapshot identity is
    either present at its exact content-addressed location or absent. There is
    no update, replace, rename, alias, activate, delete, or enumerate
    operation: :meth:`put` is idempotent, not mutable. Lifecycle concepts
    (names, channels, environments, priority, latest or active pointers,
    supersession) are out of scope.

    The store has no dependency on the profile store: a catalog may reference
    profiles that are absent, because a snapshot is a claim and not a
    population. Synchronizing the two stores is the caller's responsibility.

    Concurrency contract: concurrent puts of the same exact catalog snapshot are
    expected to converge on identical canonical content, and a new write is
    published through a same-directory temporary file plus an atomic replace.
    The store does not provide a general multi-writer transaction or locking
    protocol, and it makes no power-loss durability claim on every filesystem.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        """Create a store rooted at ``root``.

        ``root`` is caller controlled and trusted. The directory is not created
        here: :meth:`put` creates the required directories, while :meth:`get`
        reports an exact not-found without mutating the filesystem. The root may
        be the same directory used by a profile store: the two layouts are
        namespaced separately and never interpret each other's artifacts.
        """
        if not isinstance(root, (str, os.PathLike)):
            raise InvalidDecisionError(
                f"root must be a str or os.PathLike, got {type(root).__name__} ({root!r})"
            )
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The caller-supplied store root directory."""
        return self._root

    def put(self, catalog: CalibrationProfileCatalog) -> None:
        """Store ``catalog`` at its exact content-addressed path.

        The snapshot is serialized with the supported catalog serializer, so
        this store never builds the catalog wire envelope itself and there is no
        second wire-format truth source. ``put`` is idempotent: storing a
        snapshot whose exact target path already holds its canonical
        serialization is a no-op and does not rewrite the file merely to update
        timestamps. If the target path exists but is invalid or describes a
        different identity, ``put`` fails closed and never silently overwrites
        it, because existing corruption is evidence that the content-addressed
        invariant was violated.

        Referenced profiles are deliberately not required to exist: this store
        has no profile-store dependency.
        """
        if not isinstance(catalog, CalibrationProfileCatalog):
            raise InvalidDecisionError(
                f"catalog must be a CalibrationProfileCatalog, got "
                f"{type(catalog).__name__} ({catalog!r})"
            )
        fingerprint_value = catalog.fingerprint
        version = CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION
        serialized = serialize_calibration_profile_catalog(catalog)
        target = self._path_for(fingerprint_value, version)
        require_directory_root(
            self._root,
            store=_STORE_NAME,
            on_wrong_shape=_operational_error,
            on_operational=_operational_error,
        )
        reject_non_directory_components(
            _managed_components(target),
            store=_STORE_NAME,
            on_damaged=_integrity_error,
            on_operational=_operational_error,
        )
        create_managed_directories(
            target.parent, root=self._root, store=_STORE_NAME, on_operational=_operational_error
        )
        if managed_path_exists(target, store=_STORE_NAME, on_operational=_operational_error):
            self._verify_existing_artifact(target, fingerprint_value, version)
            return
        publish_atomically(
            target,
            serialized,
            temporary_prefix=".tmp-catalog-",
            store=_STORE_NAME,
            on_operational=_operational_error,
        )

    def get(
        self,
        *,
        catalog_fingerprint: str,
        catalog_fingerprint_version: int,
    ) -> CalibrationProfileCatalog:
        """Retrieve the exact catalog snapshot identified by fingerprint and version.

        Both values are required and neither is defaulted. The key is validated
        before any filesystem path is constructed, so a malformed or traversing
        key never reaches the filesystem. The requested identity is passed to
        the catalog loader as the independent expected identity pin, so the
        stored artifact must both be canonical store content and restore exactly
        the requested snapshot. An unsupported fingerprint schema version fails
        explicitly rather than being treated as an absent snapshot.

        This performs no directory scan, consults no alternate version
        directory, and applies no fallback of any kind: one key, one path, one
        result or an explicit failure.
        """
        self._validate_identity(catalog_fingerprint, catalog_fingerprint_version)
        require_directory_root(
            self._root,
            store=_STORE_NAME,
            on_wrong_shape=_operational_error,
            on_operational=_operational_error,
        )
        target = self._path_for(catalog_fingerprint, catalog_fingerprint_version)
        reject_non_directory_components(
            _managed_components(target),
            store=_STORE_NAME,
            on_damaged=_integrity_error,
            on_operational=_operational_error,
        )
        if not managed_path_exists(target, store=_STORE_NAME, on_operational=_operational_error):
            raise CalibrationProfileCatalogNotFoundError(
                f"no calibration profile catalog is stored for catalog fingerprint "
                f"{catalog_fingerprint!r} version {catalog_fingerprint_version}"
            )
        text = read_managed_text(
            target,
            artifact=_ARTIFACT_NAME,
            on_integrity=_integrity_error,
            on_operational=_operational_error,
        )
        try:
            loaded = load_calibration_profile_catalog(
                text,
                expected_catalog_fingerprint=catalog_fingerprint,
                expected_catalog_fingerprint_version=catalog_fingerprint_version,
            )
        except InvalidDecisionError as error:
            raise CalibrationProfileCatalogStoreIntegrityError(
                f"the stored calibration profile catalog at {target} is not a valid "
                f"artifact for the requested catalog fingerprint {catalog_fingerprint!r} "
                f"version {catalog_fingerprint_version}: {error}"
            ) from error
        if serialize_calibration_profile_catalog(loaded) != text:
            raise CalibrationProfileCatalogStoreIntegrityError(
                f"the stored calibration profile catalog at {target} is not the canonical "
                f"serialization for catalog fingerprint {catalog_fingerprint!r} version "
                f"{catalog_fingerprint_version}"
            )
        return loaded

    def _path_for(self, catalog_fingerprint: str, catalog_fingerprint_version: int) -> Path:
        return (
            self._root
            / _STORE_DIRECTORY
            / f"catalog-fingerprint-v{catalog_fingerprint_version}"
            / f"{catalog_fingerprint}.json"
        )

    def _validate_identity(
        self, catalog_fingerprint: object, catalog_fingerprint_version: object
    ) -> None:
        _require_catalog_fingerprint_identity(catalog_fingerprint, catalog_fingerprint_version)
        if catalog_fingerprint_version != CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION:
            raise InvalidDecisionError(
                f"the store does not understand catalog fingerprint schema version "
                f"{abbreviate_untrusted(catalog_fingerprint_version)}; it supports version "
                f"{CALIBRATION_PROFILE_CATALOG_FINGERPRINT_VERSION} only, and an unsupported "
                "identity schema is not an absent identity"
            )

    def _verify_existing_artifact(
        self,
        target: Path,
        catalog_fingerprint: str,
        catalog_fingerprint_version: int,
    ) -> None:
        text = read_managed_text(
            target,
            artifact=_ARTIFACT_NAME,
            on_integrity=_integrity_error,
            on_operational=_operational_error,
        )
        try:
            loaded = load_calibration_profile_catalog(
                text,
                expected_catalog_fingerprint=catalog_fingerprint,
                expected_catalog_fingerprint_version=catalog_fingerprint_version,
            )
        except InvalidDecisionError as error:
            raise CalibrationProfileCatalogStoreIntegrityError(
                f"a stored calibration profile catalog already exists at {target} but is "
                f"not a valid artifact for catalog fingerprint {catalog_fingerprint!r} "
                f"version {catalog_fingerprint_version}; it is not overwritten: {error}"
            ) from error
        if serialize_calibration_profile_catalog(loaded) != text:
            raise CalibrationProfileCatalogStoreIntegrityError(
                f"a stored calibration profile catalog already exists at {target} but is "
                f"not the canonical serialization for catalog fingerprint "
                f"{catalog_fingerprint!r} version {catalog_fingerprint_version}; it is not "
                "overwritten"
            )
