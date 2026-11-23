"""Bootstrap: import scikit-learn from the uid42 source tree.

The environment's editable install points at a different working copy
(/home/wll/uid191) which owns the compiled extensions. This bootstrap
registers a standard FileFinder for /home/wll/uid42 ahead of the editable
loader, so pure-Python modules resolve from uid42 while compiled (.so)
modules fall through to the editable loader's build directory.

A FileFinder (rather than a custom MetaPathFinder) keeps packages on the real
filesystem, which matters for importlib.resources data lookups used by the
test suite.

Usage: ``python3 _site_bootstrap.py <script> [args...]``
"""

import importlib.machinery
import os
import sys
import runpy

REPO = "/home/wll/uid42"

# FileFinder takes (loader_cls, suffixes) details. Source and bytecode files
# resolve from uid42; extension suffixes are deliberately omitted so compiled
# modules fall through to the editable loader's build directory.
_LOADER_DETAILS = [
    (
        importlib.machinery.SourceFileLoader,
        importlib.machinery.SOURCE_SUFFIXES,
    ),
    (
        importlib.machinery.SourcelessFileLoader,
        importlib.machinery.BYTECODE_SUFFIXES,
    ),
]

_finder = importlib.machinery.FileFinder(REPO, *_LOADER_DETAILS)


class _Uid42PathFinder:
    """Resolve sklearn* submodules from uid42 when their parent package's
    __path__ points at the uid42 tree.

    For top-level ``sklearn`` and first-level packages we consult the
    FileFinder rooted at REPO. Deeper submodules are located relative to the
    parent package's __path__ entries: once a uid42 package is in sys.modules,
    its children resolve from the same tree, even if an earlier import
    populated sklearn.utils from the editable loader before this finder ran.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sklearn" and not fullname.startswith("sklearn."):
            return None

        leaf = fullname.rsplit(".", 1)[-1]
        search_paths = []
        if path:
            search_paths.extend(path)
        if fullname in ("sklearn",) or "." not in fullname:
            search_paths.insert(0, REPO)
        if fullname.startswith("sklearn.") and "." not in fullname[len("sklearn."):]:
            search_paths.insert(0, os.path.join(REPO, "sklearn"))

        for entry in search_paths:
            # Only serve files that actually exist in the uid42 tree.
            candidate = os.path.join(entry, leaf)
            if not (
                os.path.isfile(candidate + ".py")
                or os.path.isfile(os.path.join(candidate, "__init__.py"))
            ):
                continue
            if not os.path.abspath(entry).startswith(REPO):
                continue
            finder = importlib.machinery.FileFinder(entry, *_LOADER_DETAILS)
            spec = finder.find_spec(fullname)
            if spec is not None:
                return spec

        # No uid42 source: fall through (compiled extension via the editable
        # loader, or a module absent from this tree).
        return None


sys.meta_path.insert(0, _Uid42PathFinder())
# Keep the repo on sys.path as a fallback for plain imports.
sys.path.insert(0, REPO)

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
