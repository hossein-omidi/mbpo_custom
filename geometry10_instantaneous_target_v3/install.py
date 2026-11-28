#!/usr/bin/env python3
"""Branch-independent installer for Geometry10 Instantaneous Target v3."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

PATCH_NAME = "Geometry10_Instantaneous_Target_v3.patch"
PATCH_SHA256 = "9d50eca571103a07c591bd43f5a347e1fed13f81fb962d50145bef4ab93e03a9"


class InstallError(RuntimeError):
    pass


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise InstallError("Git was not found. Install Git and retry.") from exc

    if check and result.returncode:
        message = result.stderr.strip() or result.stdout.strip()
        raise InstallError(message or f"Git command failed: {' '.join(args)}")
    return result


def find_repository(start: Path) -> Path:
    result = git(start, "rev-parse", "--show-toplevel", check=False)
    if result.returncode:
        raise InstallError(
            f"Cannot find a Git repository from: {start}\n"
            "Extract this folder inside your repository, or use --root PATH."
        )
    return Path(result.stdout.strip()).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install the standalone Geometry10 instantaneous-target v3 patch."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Check compatibility without changing files.")
    mode.add_argument("--apply", action="store_true", help="Apply the patch.")
    parser.add_argument("--root", type=Path, help="Repository path; normally detected automatically.")
    args = parser.parse_args()

    try:
        start = (args.root or Path(__file__).resolve().parent).expanduser().resolve()
        root = find_repository(start)
        patch = Path(__file__).resolve().with_name(PATCH_NAME)

        if not patch.is_file():
            raise InstallError(f"Patch file not found: {patch}")

        digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        if digest != PATCH_SHA256:
            raise InstallError(
                "Patch integrity check failed. Re-extract the original v3 installer package.\n"
                f"Expected SHA-256: {PATCH_SHA256}\nActual SHA-256:   {digest}"
            )

        # A successful reverse check means this exact patch is already present.
        already = git(
            root, "apply", "--reverse", "--check", "--whitespace=error-all",
            str(patch), check=False
        )
        if already.returncode == 0:
            print("The v3 patch is already applied. No changes made.")
            return 0

        # Compatibility is determined by the patch context, not by branch name,
        # HEAD commit, or Git status.
        check = git(
            root, "apply", "--check", "--whitespace=error-all",
            str(patch), check=False
        )
        if check.returncode:
            detail = check.stderr.strip() or check.stdout.strip()
            raise InstallError(
                "The v3 patch does not fit the current source files. No files were changed.\n"
                + (detail or "Git rejected one or more patch hunks.")
            )

        if args.dry_run:
            print("PASS: the v3 patch applies cleanly to the current working tree.")
            print("No files were changed. Run again with --apply to install.")
            return 0

        git(root, "apply", "--whitespace=error-all", str(patch))
        print("SUCCESS: v3 patch applied.")
        print(f"Repository: {root}")
        print("No branch was checked or changed. Git index and commit history were not changed.")
        return 0

    except InstallError as exc:
        print(f"INSTALLATION STOPPED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
