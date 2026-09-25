"""Private generic filesystem mechanics for exact content-addressed stores.

This module is intentionally domain neutral. It knows nothing about calibration
profiles, calibration profile catalogs, bindings, selection, or the runtime. It
implements only the generic, correctness-sensitive filesystem mechanics that the
exact profile store and the exact catalog store must share so the two can never
drift apart on them:

* directory-root shape checking,
* managed-layout corruption classification,
* strict UTF-8 reads, and
* same-directory temporary publication with ``fsync`` and an atomic replace.

Shape probing never uses ``Path.exists`` / ``Path.is_dir`` directly: those
suppress only the absence errnos and re-raise the rest, so an unreadable parent
directory would leak a raw ``PermissionError``. Every probe maps such an
``OSError`` through the caller's operational failure builder instead.

Every function receives the caller's own exception builder, so each store keeps
its own error taxonomy and message vocabulary. There is deliberately no store
class, no protocol, and no registry here: a shared abstraction over artifact
identity or validation would be a second source of truth for the store
invariants, which is exactly what must not exist.

The functions accept the two noun phrases a caller needs to phrase its own
messages: the singular artifact name (for example ``"calibration profile"``) and
the store name (for example ``"calibration profile store"``). No other domain
knowledge is accepted or assumed.
"""

from __future__ import annotations

import contextlib
import errno
import os
import stat
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

#: Builds the caller's own exception from a fully formed detail message.
FailureBuilder = Callable[[str], Exception]

#: ``errno`` values that mean "this path does not exist", not "I could not
#: look". They mirror the set that :mod:`pathlib` itself suppresses in
#: ``Path.exists``: a missing entry, a non-directory on the way, a bad
#: descriptor, or a symlink loop. Anything else, notably ``EACCES``, is a
#: managed-store access failure that must not be mistaken for absence.
_ABSENCE_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP})

#: What a managed path currently is on disk.
ManagedPathState = Literal["absent", "directory", "other"]


def managed_path_state(
    path: Path,
    *,
    store: str,
    on_operational: FailureBuilder,
) -> ManagedPathState:
    """Classify ``path`` as ``"absent"``, ``"directory"``, or ``"other"``.

    ``Path.exists`` and ``Path.is_dir`` suppress only the absence errnos and
    re-raise everything else, so an unreadable parent directory would leak a raw
    ``PermissionError`` out of the public store API instead of the store's own
    operational error. Probing through :func:`os.stat` and mapping every other
    ``OSError`` through ``on_operational`` closes that leak.
    """
    try:
        mode = os.stat(path).st_mode
    except OSError as error:
        if error.errno in _ABSENCE_ERRNOS:
            return "absent"
        raise on_operational(f"the {store} could not inspect {path}: {error}") from error
    return "directory" if stat.S_ISDIR(mode) else "other"


def managed_path_exists(
    path: Path,
    *,
    store: str,
    on_operational: FailureBuilder,
) -> bool:
    """Report whether ``path`` exists, without leaking a raw ``OSError``."""
    return managed_path_state(path, store=store, on_operational=on_operational) != "absent"


def require_directory_root(
    root: Path,
    *,
    store: str,
    on_wrong_shape: FailureBuilder,
    on_operational: FailureBuilder,
) -> None:
    """Reject a store root that exists but is not a directory.

    An absent root is not a failure: it simply means nothing is stored yet.
    A root that exists as a different filesystem object is an operational
    failure of the managed store, not ordinary absence.
    """
    if managed_path_state(root, store=store, on_operational=on_operational) == "other":
        raise on_wrong_shape(f"the {store} root is not a directory: {root}")


def reject_non_directory_components(
    components: Sequence[tuple[Path, str]],
    *,
    store: str,
    on_damaged: FailureBuilder,
    on_operational: FailureBuilder,
) -> None:
    """Reject a managed layout component that exists but is not a directory.

    An absent managed directory means the exact artifact is simply absent. A
    component that exists as an incompatible filesystem object means the
    managed store layout was damaged, which is corruption and never ordinary
    absence.
    """
    for path, role in components:
        state = managed_path_state(path, store=store, on_operational=on_operational)
        if state == "other":
            raise on_damaged(
                f"the managed {store} {role} exists but is not a directory, so the "
                f"store layout is damaged: {path}"
            )


def create_managed_directories(
    directories: Path,
    *,
    root: Path,
    store: str,
    on_operational: FailureBuilder,
) -> None:
    """Create a store's managed parent directories if they are absent."""
    try:
        directories.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise on_operational(
            f"the {store} could not create its directories under {root}: {error}"
        ) from error


def read_managed_text(
    target: Path,
    *,
    artifact: str,
    on_integrity: FailureBuilder,
    on_operational: FailureBuilder,
) -> str:
    """Read a managed artifact as strict UTF-8.

    Invalid UTF-8 at an exact artifact path is corruption of a managed artifact,
    not absence, so it is reported through ``on_integrity``. Any other
    filesystem failure is operational.
    """
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise on_integrity(
            f"the stored {artifact} at {target} is not valid UTF-8: {error}"
        ) from error
    except OSError as error:
        raise on_operational(
            f"the stored {artifact} at {target} could not be read: {error}"
        ) from error


def publish_atomically(
    target: Path,
    serialized: str,
    *,
    temporary_prefix: str,
    store: str,
    on_operational: FailureBuilder,
) -> None:
    """Publish canonical text at ``target`` through a temporary file.

    The text is written to a temporary file in the target directory, flushed and
    ``fsync``-ed, then moved into place with :func:`os.replace`, so the final
    path never exposes a partially written artifact. A controlled failure
    removes the temporary file.

    The same-filesystem replace is atomic under this primitive. This makes no
    universal power-loss durability claim, no multi-host transaction claim, and
    no adversarial multi-writer claim.
    """
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent, prefix=temporary_prefix, suffix=".json"
        )
    except OSError as error:
        raise on_operational(
            f"the {store} could not create a temporary file next to {target}: {error}"
        ) from error
    temporary_path = Path(temporary_name)
    try:
        try:
            handle = os.fdopen(descriptor, "w", encoding="utf-8")
        except OSError as error:
            # ``os.fdopen`` failed before taking ownership of the descriptor, so
            # close it here or it leaks.
            with contextlib.suppress(OSError):
                os.close(descriptor)
            raise on_operational(
                f"the {store} could not publish the artifact at {target}: {error}"
            ) from error
        with handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except OSError as error:
        raise on_operational(
            f"the {store} could not publish the artifact at {target}: {error}"
        ) from error
    finally:
        with contextlib.suppress(OSError):
            temporary_path.unlink()
