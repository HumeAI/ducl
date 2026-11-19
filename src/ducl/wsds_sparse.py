"""Correct scan sizes of wsds sparse-cache mirrors (``*.wsds.sparse``).

WEKA reports ``st_blocks`` as if a file were dense, so a sparse mirror's stat
carries its apparent size -- the full object length plus bitmap and footer --
no matter how few blocks were ever fetched. The mirror file is self-describing
(wsds ``pupyarrow/block_cache.py``): 128 KiB data slots for the whole object,
then a one-byte-per-block presence bitmap, then a footer of magic + object
size. The resident footprint is therefore computable from the file alone:
present data blocks (the last one counts only the object tail) plus the
bitmap/footer metadata that every mirror physically stores.

``SparseSizeFixer`` patches ``size`` and ``blocks`` of matching rows in scan
batches before they reach the feather writer and the agg sidecars, so every
downstream consumer (dashboard, diff, examples) sees resident bytes. The
walker-computed ``dir_total_size`` column on directory rows is NOT corrected
and still sums apparent sizes; the dashboards do not read it (they re-sum
file rows), but an rsync-style differ leaning on it would see the stat view.

A mirror that cannot be read or fails the magic check keeps its stat size.
"""

from __future__ import annotations

import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

SLOT = 128 * 1024
FOOTER = struct.Struct("<8sQ")  # magic, object_size
MAGIC = b"WKSPRS1\x00"
FOOT_SZ = FOOTER.size
SUFFIX = ".wsds.sparse"

_READERS = 32  # concurrent bitmap reads; each is two small preads


def resident_size(path: str) -> int | None:
    """Resident bytes of one mirror, or None if unreadable or not a mirror."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        end = os.fstat(fd).st_size
        if end < FOOT_SZ:
            return None
        magic, objsize = FOOTER.unpack(os.pread(fd, FOOT_SZ, end - FOOT_SZ))
        if magic != MAGIC:
            return None
        nblocks = (objsize + SLOT - 1) // SLOT
        bm = os.pread(fd, nblocks, nblocks * SLOT)
        if len(bm) != nblocks:
            return None
        present = nblocks - bm.count(0)
        size = present * SLOT
        if present and bm[-1]:  # last block holds only the object tail
            size -= SLOT - (objsize - (nblocks - 1) * SLOT)
        return size + nblocks + FOOT_SZ  # bitmap + footer are always resident
    except OSError:
        return None
    finally:
        os.close(fd)


class SparseSizeFixer:
    """Rewrite size/blocks of ``*.wsds.sparse`` rows in scan RecordBatches."""

    def __init__(self) -> None:
        self.n_fixed = 0
        self.n_failed = 0
        self.apparent = 0
        self.resident = 0

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        pi = batch.schema.get_field_index("path")
        mask = pc.ends_with(batch.column(pi), pattern=SUFFIX)
        if pc.sum(mask).as_py() in (None, 0):
            return batch

        idx = np.flatnonzero(mask.to_numpy(zero_copy_only=False))
        paths = batch.column(pi).take(pa.array(idx)).to_pylist()
        si = batch.schema.get_field_index("size")
        bi = batch.schema.get_field_index("blocks")
        sizes = batch.column(si).to_numpy(zero_copy_only=False).copy()
        blocks = batch.column(bi).to_numpy(zero_copy_only=False).copy()

        with ThreadPoolExecutor(min(_READERS, len(idx))) as pool:
            for i, res in zip(idx, pool.map(resident_size, paths)):
                if res is None:
                    self.n_failed += 1
                    continue
                self.apparent += int(sizes[i])
                self.resident += res
                sizes[i] = res
                blocks[i] = (res + 511) // 512
                self.n_fixed += 1

        arrays = list(batch.columns)
        arrays[si] = pa.array(sizes, type=batch.schema.field(si).type)
        arrays[bi] = pa.array(blocks, type=batch.schema.field(bi).type)
        return pa.RecordBatch.from_arrays(arrays, schema=batch.schema)

    def report(self) -> None:
        if self.n_fixed or self.n_failed:
            print(
                f"wsds sparse mirrors: {self.n_fixed} resized, "
                f"{self.apparent / 1e9:.1f} GB apparent -> "
                f"{self.resident / 1e9:.1f} GB resident"
                + (f" ({self.n_failed} unreadable, kept stat size)"
                   if self.n_failed else ""),
                file=sys.stderr,
            )
