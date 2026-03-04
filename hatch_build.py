"""Hatchling build hook -- compiles pwalk2 from bundled C sources.

On Linux the hook compiles the bundled pwalk2 C sources and marks the
wheel as platform-specific so pip installs the right binary per arch.
On non-Linux the hook is a no-op and the wheel stays ``py3-none-any``.
"""

import os
import platform
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

        subprocess.check_call(["make", "-C", pw2_dir, "pwalk2"])

        binary = os.path.join(pw2_dir, "pwalk2")
        if not os.path.isfile(binary):
            return

        # Include compiled binary in wheel
        build_data.setdefault("force_include", {})[binary] = "ducl/pwalk2/pwalk2"

        # Mark wheel as platform-specific (not pure-python)
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        # Override platform tag to a manylinux tag so PyPI accepts it
        machine = platform.machine()
        plat_tag = _MACHINE_TO_PLAT.get(machine)
        if plat_tag:
            build_data["tag"] = f"py3-none-{plat_tag}"
