"""Hatchling build hook -- compiles pwalk2 from bundled C sources.

On Linux the hook compiles the bundled pwalk2 C sources and marks the
wheel as platform-specific so pip installs the right binary per arch.
On non-Linux the hook is a no-op and the wheel stays ``py3-none-any``.

Toolchain: if ``zig`` is on PATH or the ``ziglang`` PyPI package is
installed, the Makefile cross-compiles against glibc 2.17 and the wheel is
tagged ``manylinux_2_17_<arch>`` (runs on any distro since CentOS 7).
Otherwise it falls back to the host gcc; the binary then needs the build
host's glibc or newer, so the wheel is tagged plain ``linux_<arch>`` to
avoid advertising portability it does not have.
"""

import importlib.util
import os
import platform
import shutil
import subprocess

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# Map Python's platform.machine() to manylinux platform tags
_MACHINE_TO_PLAT = {
    "x86_64":  "manylinux_2_17_x86_64",
    "aarch64": "manylinux_2_17_aarch64",
}


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if platform.system() != "Linux":
            return  # pwalk2 is Linux-only; skip silently

        pw2_dir = os.path.join(self.root, "src", "ducl", "pwalk2")
        if not os.path.isdir(pw2_dir):
            return

        # Mirrors the Makefile's detection. `python-zig` is the console
        # script of the ziglang package; checking PATH for it also works
        # from inside an isolated `python -m build` venv.
        have_zig = (shutil.which("zig") is not None
                    or shutil.which("python-zig") is not None
                    or importlib.util.find_spec("ziglang") is not None)
        if not have_zig:
            print("hatch_build: zig not found; building pwalk2 with host gcc "
                  "(wheel will not be manylinux). `pip install ziglang` to fix.")
        subprocess.check_call(["make", "-C", pw2_dir, "-B", "pwalk2"])

        binary = os.path.join(pw2_dir, "pwalk2")
        if not os.path.isfile(binary):
            return

        # Include compiled binary in wheel
        build_data.setdefault("force_include", {})[binary] = "ducl/pwalk2/pwalk2"

        # Mark wheel as platform-specific (not pure-python)
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        # Platform tag: manylinux_2_17 only when the binary really was built
        # against glibc 2.17 (zig); otherwise an honest non-manylinux tag.
        machine = platform.machine()
        plat_tag = _MACHINE_TO_PLAT.get(machine) if have_zig else None
        if plat_tag is None:
            plat_tag = f"linux_{machine}"
        build_data["tag"] = f"py3-none-{plat_tag}"
