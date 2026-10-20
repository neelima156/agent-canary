"""Per-run working folder. Writes cannot leave it.

Same policy as agent.safe: only the file's base name is kept, so a path
like ../../CANARY.txt still lands inside the sandbox. The check uses
commonpath so a prefix of the sandbox path is not treated as inside it.
"""
import os
import shutil


def basename_only(path):
    name = str(path).replace("\\", "/").split("/")[-1].strip()
    if not name or name in (".", "..") or "\x00" in name:
        raise ValueError("outside sandbox")
    return name


def sandbox_path(sandbox, path):
    name = basename_only(path)
    root = os.path.realpath(sandbox)
    full = os.path.realpath(os.path.join(root, name))
    try:
        shared = os.path.commonpath([root, full])
    except ValueError:
        raise ValueError("outside sandbox") from None
    if shared != root:
        raise ValueError("outside sandbox")
    return full


def reset_sandbox(path):
    """Replace path with a fresh real directory. A pre-planted symlink is unlinked."""
    if os.path.islink(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)
    os.makedirs(path)
    return path
