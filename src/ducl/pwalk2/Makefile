# pwalk2 build
#
# Toolchain selection (in order of preference):
#   1. CC given explicitly on the command line / environment  (make CC=...)
#   2. zig cc cross-compiling against glibc $(GLIBC_MIN) -- produces a binary
#      that runs on any distro with glibc >= 2.17 (CentOS 7 / manylinux_2_17),
#      regardless of the glibc on the build host.  Found via `zig` on PATH,
#      the `python-zig` script installed by the `ziglang` PyPI package
#      (pip install ziglang), or `python3 -m ziglang`.
#   3. Plain host gcc -- the binary then needs the build host's glibc or newer.
#
# Override with e.g.  make ZIG=/opt/zig/zig  or  make GLIBC_MIN=2.28 ARCH=aarch64

ARCH      ?= $(shell uname -m)
GLIBC_MIN ?= 2.17
ZIG       ?= $(shell command -v zig 2>/dev/null || command -v python-zig 2>/dev/null || \
               (python3 -c 'import ziglang' 2>/dev/null && echo 'python3 -m ziglang'))

ifeq ($(origin CC),default)
  ifneq ($(strip $(ZIG)),)
    CC = $(ZIG) cc -target $(ARCH)-linux-gnu.$(GLIBC_MIN)
  else
    CC = gcc
  endif
endif

CFLAGS  = -O2 -Wall -std=gnu11
LDFLAGS = -lpthread

pwalk2: pwalk2.c pw2_worker.c pw2_output.c pwalk2.h pw2_uring.h
	@echo "CC = $(CC)"
	$(CC) $(CFLAGS) $(LDFLAGS) -o pwalk2 pwalk2.c pw2_worker.c pw2_output.c

# Print the toolchain the build will use (consumed by hatch_build.py)
print-cc:
	@echo "$(CC)"

clean:
	rm -f pwalk2

.PHONY: clean print-cc
