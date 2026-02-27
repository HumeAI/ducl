"""Hatchling build hook -- compiles pwalk2 from bundled C sources."""

import os
import platform
import subprocess

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if platform.system() != "Linux":
            return  # pwalk2 is Linux-only; skip silently
        pw2_dir = os.path.join(self.root, "src", "ducl", "pwalk2")
        if not os.path.isdir(pw2_dir):
            return
        subprocess.check_call(["make", "-C", pw2_dir, "pwalk2"])
        # Include compiled binary in wheel
        binary = os.path.join(pw2_dir, "pwalk2")
        if os.path.isfile(binary):
            build_data.setdefault("force-include", {})[binary] = "ducl/pwalk2/pwalk2"
