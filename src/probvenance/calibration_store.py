"""Exact content-addressed directory store for calibration profiles.

This module implements the first supported persistent profile store. It answers
exactly one question: given an exact profile fingerprint and fingerprint
schema version, retrieve that exact profile artifact.

It deliberately answers no selection-policy question. There is no binding
lookup, no method or taxonomy matching, no nearest or singleton fallback, no
latest or best profile, and no enumeration API. A directory containing
profiles is not a semantic registry: the caller must already know the exact
identity it wants.

Dependency direction: this module depends on :mod:`probvenance.calibration` for
the serialization contract and on the fingerprint/error foundations. The
calibration, runtime, results, and trace modules never import this one.

The store is content addressed. A profile's path is derived only from its
fingerprint schema version and its exact fingerprint, and the stored bytes are
exactly the canonical serialization emitted by
:func:`probvenance.calibration.serialize_calibration_profile`. Retrieval
supplies the requested identity to the loader as an independent expected
identity pin, so a different but internally self-consistent profile placed at
the requested path is rejected.
"""

import contextlib
import os
import re
import tempfile
from pathlib import Path

from probvenance.calibration import (
    CALIBRATION_PROFILE_FINGERPRINT_VERSION,
    CalibrationProfile,
    _abbreviate,
    load_calibration_profile,
    serialize_calibration_profile,
)
from probvenance.errors import (
    CalibrationProfileNotFoundError,
    CalibrationProfileStoreError,
    CalibrationProfileStoreIntegrityError,
    InvalidDecisionError,
)

CALIBRATION_PROFILE_DIRECTORY_STORE_VERSION = 1
"""Version of the on-disk directory layout.

This is independent of the profile fingerprint schema version, the profile
serialization version, the calibration method version, and the solver version.
None of these versions are interchangeable.
"""

_STORE_DIRECTORY = f"store-v{CALIBRATION_PROFILE_DIRECTORY_STORE_VERSION}"
_FINGERPRINT_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")


class DirectoryCalibrationProfileStore:
    """Exact, content-addressed, directory-backed calibration profile store.

    The store is immutable in the sense that a profile identity is either
    present at its exact content-addressed location or absent. There is no
    update, replace, rename, alias, or delete operation: :meth:`put` is
    idempotent, not mutable. Selection and lifecycle concepts (names, tags,
    channels, priority, environment) are out of scope.

    Concurrency contract: concurrent puts of the same exact profile are
    expected to converge on identical canonical content, and a new write is
    published through a same-directory temporary file plus an atomic
    replace. The store does not provide a general multi-writer transaction or
    locking protocol, and it makes no power-loss durability claim on every
    filesystem.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        """Create a store rooted at ``root``.

        ``root`` is caller controlled and trusted. The directory is not created
        here: :meth:`put` creates the required directories, while :meth:`get`
        reports an exact not-found without mutating the filesystem.
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

    def put(self, profile: CalibrationProfile) -> None:
        """Store ``profile`` at its exact content-addressed path.

        The profile is serialized with the supported serializer, so this store
        never builds the wire envelope itself. ``put`` is idempotent: storing a
        profile whose exact target path already holds its canonical
        serialization is a no-op. If the target path exists but is invalid or
        describes a different identity, ``put`` fails closed and never
        silently overwrites it, because existing corruption is evidence that
        the content-addressed invariant was violated.
        """
        if not isinstance(profile, CalibrationProfile):
            raise InvalidDecisionError(
                f"profile must be a CalibrationProfile, got {type(profile).__name__} ({profile!r})"
            )
        fingerprint = profile.fingerprint
        version = CALIBRATION_PROFILE_FINGERPRINT_VERSION
        serialized = serialize_calibration_profile(profile)
        target = self._path_for(fingerprint, version)
        self._require_usable_root()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise CalibrationProfileStoreError(
                f"the calibration profile store could not create its directories "
                f"under {self._root}: {error}"
            ) from error
        if target.exists():
            self._verify_existing_artifact(target, fingerprint, version)
            return
        self._write_atomically(target, serialized)

    def get(
        self,
        *,
        profile_fingerprint: str,
        profile_fingerprint_version: int,
    ) -> CalibrationProfile:
        """Retrieve the exact profile identified by fingerprint and version.

        Both values are required and neither is defaulted. The requested
        identity is passed to the loader as the independent expected identity
        pin, so the stored artifact must both be canonical store content and
        describe exactly the requested profile. This performs no directory
        scan and no fallback of any kind: one key, one path, one result or an
        explicit failure.
        """
        self._validate_identity(profile_fingerprint, profile_fingerprint_version)
        self._require_usable_root()
        target = self._path_for(profile_fingerprint, profile_fingerprint_version)
        if not target.exists():
            raise CalibrationProfileNotFoundError(
                f"no calibration profile is stored for profile fingerprint "
                f"{profile_fingerprint!r} version {profile_fingerprint_version}"
            )
        text = self._read_text(target)
        try:
            loaded = load_calibration_profile(
                text,
                expected_profile_fingerprint=profile_fingerprint,
                expected_profile_fingerprint_version=profile_fingerprint_version,
            )
        except InvalidDecisionError as error:
            raise CalibrationProfileStoreIntegrityError(
                f"the stored calibration profile at {target} is not a valid artifact "
                f"for the requested profile fingerprint {profile_fingerprint!r} version "
                f"{profile_fingerprint_version}: {error}"
            ) from error
        if serialize_calibration_profile(loaded) != text:
            raise CalibrationProfileStoreIntegrityError(
                f"the stored calibration profile at {target} is not the canonical "
                f"serialization for profile fingerprint {profile_fingerprint!r} version "
                f"{profile_fingerprint_version}"
            )
        return loaded

    def _path_for(self, profile_fingerprint: str, profile_fingerprint_version: int) -> Path:
        return (
            self._root
            / _STORE_DIRECTORY
            / f"profile-fingerprint-v{profile_fingerprint_version}"
            / f"{profile_fingerprint}.json"
        )

    def _validate_identity(
        self, profile_fingerprint: object, profile_fingerprint_version: object
    ) -> None:
        if not isinstance(profile_fingerprint, str) or not _FINGERPRINT_PATTERN.match(
            profile_fingerprint
        ):
            raise InvalidDecisionError(
                "profile_fingerprint must be exactly 64 lowercase hexadecimal "
                f"characters, got {_abbreviate(profile_fingerprint)}"
            )
        if (
            isinstance(profile_fingerprint_version, bool)
            or not isinstance(profile_fingerprint_version, int)
            or profile_fingerprint_version < 1
        ):
            raise InvalidDecisionError(
                "profile_fingerprint_version must be an integer greater than or equal "
                f"to 1, got {profile_fingerprint_version!r}"
            )
        if profile_fingerprint_version != CALIBRATION_PROFILE_FINGERPRINT_VERSION:
            raise InvalidDecisionError(
                f"the store does not understand profile fingerprint schema version "
                f"{profile_fingerprint_version}; it supports version "
                f"{CALIBRATION_PROFILE_FINGERPRINT_VERSION} only, and an unsupported "
                "identity schema is not an absent identity"
            )

    def _require_usable_root(self) -> None:
        if self._root.exists() and not self._root.is_dir():
            raise CalibrationProfileStoreError(
                f"the calibration profile store root is not a directory: {self._root}"
            )

    def _read_text(self, target: Path) -> str:
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise CalibrationProfileStoreIntegrityError(
                f"the stored calibration profile at {target} is not valid UTF-8: {error}"
            ) from error
        except OSError as error:
            raise CalibrationProfileStoreError(
                f"the stored calibration profile at {target} could not be read: {error}"
            ) from error

    def _verify_existing_artifact(
        self, target: Path, profile_fingerprint: str, profile_fingerprint_version: int
    ) -> None:
        text = self._read_text(target)
        try:
            loaded = load_calibration_profile(
                text,
                expected_profile_fingerprint=profile_fingerprint,
                expected_profile_fingerprint_version=profile_fingerprint_version,
            )
        except InvalidDecisionError as error:
            raise CalibrationProfileStoreIntegrityError(
                f"a stored calibration profile already exists at {target} but is not a "
                f"valid artifact for profile fingerprint {profile_fingerprint!r} version "
                f"{profile_fingerprint_version}; it is not overwritten: {error}"
            ) from error
        if serialize_calibration_profile(loaded) != text:
            raise CalibrationProfileStoreIntegrityError(
                f"a stored calibration profile already exists at {target} but is not the "
                f"canonical serialization for profile fingerprint {profile_fingerprint!r} "
                f"version {profile_fingerprint_version}; it is not overwritten"
            )

    def _write_atomically(self, target: Path, serialized: str) -> None:
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=target.parent, prefix=".tmp-profile-", suffix=".json"
            )
        except OSError as error:
            raise CalibrationProfileStoreError(
                f"the calibration profile store could not create a temporary file "
                f"next to {target}: {error}"
            ) from error
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
        except OSError as error:
            raise CalibrationProfileStoreError(
                f"the calibration profile store could not publish the artifact at {target}: {error}"
            ) from error
        finally:
            if temporary_path.exists():
                with contextlib.suppress(OSError):
                    temporary_path.unlink()
