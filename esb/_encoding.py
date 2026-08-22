"""UTF-8 console output, so no caller has to prefix PYTHONIOENCODING.

The pipeline prints non-ASCII -- the ``Rows: N -> M`` arrow in
``deletingMissingValues`` (utils_mse_crps.py), a warning sign beside it, em
dashes in several ``esb`` banners. Windows selects cp1252 for a REDIRECTED
stream, so ``... > log`` or ``... | tee log`` raises UnicodeEncodeError on the
first such print -- typically deep into a run that has already burned hours.

Call ``force_utf8()`` at the top of every entry point. It is deliberately an
explicit call rather than an import side effect in ``esb/__init__.py``: a
process quietly rewriting its own stdout encoding is the kind of magic that is
impossible to find later when you want to know why output looks different.
"""

from __future__ import annotations

import os
import sys


def force_utf8() -> None:
    """Make this process and its future children write UTF-8.

    Two settings are needed, and neither substitutes for the other:

    * ``reconfigure()`` fixes THIS process. Its streams already exist by the
      time any environment variable we set could be consulted.
    * ``PYTHONIOENCODING`` fixes SPAWNED WORKERS, which inherit ``os.environ``
      and construct their own streams at startup. Most pipeline output comes
      from workers, so reconfigure alone leaves the original problem in place.

    ``setdefault`` so an explicitly-exported PYTHONIOENCODING still wins, and
    ``errors="replace"`` so a stray character degrades to a placeholder instead
    of killing a long run.

    Safe to call more than once.
    """
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            # Already-wrapped (pytest capture, a pipe object) or otherwise not
            # reconfigurable. Not fatal: the env var still covers children.
            pass
