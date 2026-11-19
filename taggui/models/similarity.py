"""
Perceptual similarity of the loaded images (the Find Similar Images
feature): 256-bit pHash computation, vectorized Hamming distances, and
seed-based (non-transitive) grouping.

Imported only by the Find Similar Images dialog, which the main window
imports lazily when the dialog is opened, so the heavy dependencies are
never loaded at application start: `numpy` is imported at module level here
(moderate cost, paid once on dialog open) and `imagehash`/`scipy` are
imported inside `_load_for_phash` (paid on the first scan, in the worker
thread).

The 256-bit hashes are stored in the database as 64-character hex strings
(the `phash` table, schema version 4 — a 256-bit value does not fit in a
SQLite `INTEGER`) and compared here as 32 bytes through a lookup-table
Hamming distance: numpy has no popcount ufunc, so the distance of two bytes
is `_HAMMING_LUT[x ^ y]`, accumulated over the 32 bytes of a hash. The
LUT/`XOR`/accumulate loop releases the GIL on the numpy calls, so a
`ThreadPoolExecutor` parallelizes it across cores (measured ~10x with 16
workers on this machine).
"""

import threading
from pathlib import Path

import numpy as np

# The pHash parameters, in one place so the stored `phash_version` always
# matches the computation: imagehash 4.x `phash` with `hash_size=16`
# (16x16 bits = 256 bits); the image is exif-transposed, its alpha channel
# composited over white, converted to RGB, and (for JPEG) decoded at most
# 128x128 via `draft`; the first frame of an animated image.
PHASH_BITS = 256
PHASH_BYTES = PHASH_BITS // 8
PHASH_VERSION = 1

# How far to decode JPEGs for the hash (the `draft` size hint).
_JPEG_DRAFT_SIZE = (128, 128)

# A generous upper bound for decompression-bomb protection, raised for the
# scan only (it runs in worker threads); legitimate huge images (e.g.
# 200MP scans) must still be hashable.
_MAX_PHASH_PIXELS = 400_000_000

# Row chunk of the distance pass. With `worker_count` chunks in flight at
# once, the in-flight distance memory is ~5*chunk*N bytes (the uint16
# result matrix plus a uint8 XOR and a uint16 LUT temporary per byte
# column): at the default 16 workers that is ~0.5 GiB for N=50k with
# chunk=128 (vs ~2 GiB for chunk=512), for the same total work.
_DISTANCE_CHUNK = 128

# Popcount of every possible byte value, for the lookup-table Hamming
# distance.
_HAMMING_LUT = np.array([bin(i).count('1') for i in range(256)],
                        dtype=np.uint16)


class ScanCancelled(Exception):
    """Raised by `compute_neighbors` when the scan is cancelled."""


class TooManyMatches(Exception):
    """Raised by `compute_neighbors` when more than `max_pairs` neighbor
    links would be stored (a threshold so broad that nearly everything
    matches everything). The caller should lower the threshold."""


def _load_for_phash(image_path: Path, mirrored: bool):
    """
    The image at `image_path`, prepared for pHash (EXIF orientation
    honored, first frame, alpha over white, RGB, JPEGs at most
    128x128), or None when it cannot be read.

    Left-right mirrored when `mirrored` (the optional mirror matching).
    The returned image is left open: the caller hashes it and closes it.
    """
    from PIL import Image, ImageOps

    try:
        previous_limit = Image.MAX_IMAGE_PIXELS
        if not previous_limit or previous_limit < _MAX_PHASH_PIXELS:
            Image.MAX_IMAGE_PIXELS = _MAX_PHASH_PIXELS
    except AttributeError:
        pass
    try:
        with Image.open(image_path) as image:
            # The first frame of an animated image.
            image.seek(0)
            # Honor the EXIF orientation (the same treatment as the
            # thumbnail and captioning load paths). When no orientation is
            # stored this returns the image itself, without loading it.
            image = ImageOps.exif_transpose(image)
            if image is not None and image.format == 'JPEG':
                # Decode at a reduced size (a no-op if the image was
                # already loaded, e.g. by a real orientation transpose).
                image.draft('RGB', _JPEG_DRAFT_SIZE)
            if mirrored:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if image.mode in ('RGBA', 'LA', 'PA') or (
                    image.mode == 'P' and 'transparency' in image.info):
                # An alpha channel: composite over white so transparent
                # pixels do not hash as black.
                background = Image.new('RGB', image.size, (255, 255, 255))
                rgba = image.convert('RGBA')
                background.paste(rgba, mask=rgba.split()[-1])
                image = background
            else:
                image = image.convert('RGB')
            # A copy: the `with` block closes the opened file, but the
            # (possibly transposed/composited) image must stay readable.
            return image.copy()
    except Exception:
        return None
    finally:
        try:
            Image.MAX_IMAGE_PIXELS = previous_limit
        except (AttributeError, NameError):
            pass


def compute_phash(image_path: Path) -> int | None:
    """
    The 256-bit pHash of the image at `image_path`, as an integer, or None
    when it cannot be computed (corrupt file, unreadable format, ...).

    A failed hash must never fail the scan: the caller counts the image as
    skipped.
    """
    image = _load_for_phash(image_path, mirrored=False)
    if image is None:
        return None
    try:
        import imagehash
        return _imagehash_to_int(imagehash.phash(image, hash_size=16))
    except Exception:
        return None
    finally:
        image.close()


def compute_phash_mirrored(image_path: Path) -> int | None:
    """
    The 256-bit pHash of the left-right mirrored image at `image_path`
    (the optional mirror matching), or None when it cannot be computed.
    """
    image = _load_for_phash(image_path, mirrored=True)
    if image is None:
        return None
    try:
        import imagehash
        return _imagehash_to_int(imagehash.phash(image, hash_size=16))
    except Exception:
        return None
    finally:
        image.close()


def _imagehash_to_int(image_hash) -> int:
    """A 256-bit `imagehash.ImageHash` as a Python int (row-major bits,
    bit-packed: the first element is the most significant bit)."""
    bits = image_hash.hash.flatten().astype(np.uint8)
    packed = np.packbits(bits, bitorder='big')
    return int.from_bytes(packed.tobytes(), 'big')


def phash_to_hex(value: int) -> str:
    """The 64-character hex string of a 256-bit hash (the DB form)."""
    return value.to_bytes(PHASH_BYTES, 'big').hex()


def phash_from_hex(text: str) -> int:
    """The int form of a stored 64-character hex hash."""
    return int.from_bytes(bytes.fromhex(text), 'big')


def hashes_to_bytes(hashes: list[int]) -> np.ndarray:
    """The (N, 32) uint8 matrix of the given 256-bit hashes."""
    raw = b''.join(value.to_bytes(PHASH_BYTES, 'big') for value in hashes)
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(hashes),
                                                      PHASH_BYTES).copy()


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """The Hamming distance (0..256) of two 256-bit hashes, as integers."""
    raw = (hash_a ^ hash_b).to_bytes(PHASH_BYTES, 'big')
    return int(_HAMMING_LUT[np.frombuffer(raw, dtype=np.uint8)].sum())


def _chunk_distances(chunk: np.ndarray, all_bytes: np.ndarray) -> np.ndarray:
    """
    The (len(chunk), len(all_bytes)) uint16 Hamming distance matrix between
    the chunk's hashes and all the hashes.
    """
    distances = np.zeros((chunk.shape[0], all_bytes.shape[0]),
                         dtype=np.uint16)
    for k in range(PHASH_BYTES):
        distances += _HAMMING_LUT[chunk[:, k, None] ^ all_bytes[None, :, k]]
    return distances


def compute_neighbors(hashes: list[int], threshold: int,
                      mirrored_hashes: list[int] | None = None,
                      excluded: set[tuple[int, int]] | None = None,
                      worker_count: int = 8,
                      should_cancel=None,
                      max_pairs: int | None = None) -> list:
    """
    For every hash, the list of (index, distance, mirrored) of the other
    hashes within `threshold` of it (index != its own).

    `mirrored_hashes`, when given (the mirror-matching option), is the
    256-bit pHash of each image's left-right mirrored version: the pair
    distance is `min(direct, mirrored)` and the `mirrored` flag says which
    was smaller.

    `excluded` is a set of (min(a, b), max(a, b)) index pairs whose links
    are dropped (ignored and split pairs).

    The distance pass runs in a thread pool (the numpy calls release the
    GIL); it checks `should_cancel()` between batches of chunks and raises
    `ScanCancelled` (with a partial result) when it fires.

    When `max_pairs` is given, it raises `TooManyMatches` once the total
    number of stored links passes that count — a guard against a threshold
    so broad that nearly everything matches everything (e.g. random images
    at a high threshold), whose neighbor lists would be unbounded.
    """
    from concurrent.futures import ThreadPoolExecutor

    all_bytes = hashes_to_bytes(hashes)
    mirror_bytes = (hashes_to_bytes(mirrored_hashes)
                    if mirrored_hashes is not None else None)
    count = len(hashes)
    excluded = excluded or set()
    neighbors: list = [[] for _ in range(count)]
    starts = list(range(0, count, _DISTANCE_CHUNK))
    # Shared across the pool threads: the total number of stored links,
    # guarded by the cap below.
    pair_count = [0]
    pair_count_lock = threading.Lock()

    def work(start: int):
        end = min(start + _DISTANCE_CHUNK, count)
        direct = _chunk_distances(all_bytes[start:end], all_bytes)
        if mirror_bytes is not None:
            mirrored = _chunk_distances(all_bytes[start:end], mirror_bytes)
            mask = mirrored < direct
            np.minimum(direct, mirrored, out=direct)
        else:
            mask = None
        for row in range(end - start):
            i = start + row
            # The self pair (distance 0) is not a neighbor.
            row_distances = direct[row]
            row_distances[i] = PHASH_BITS + 1
            if mask is not None:
                row_mask = mask[row]
                row_mask[i] = False
            indices = np.nonzero(row_distances <= threshold)[0]
            row_neighbours = neighbors[i]
            for j in indices.tolist():
                pair = (i, j) if i < j else (j, i)
                if pair in excluded:
                    continue
                row_neighbours.append(
                    (int(j), int(row_distances[j]),
                     bool(row_mask[j]) if mask is not None else False))
        if max_pairs is not None:
            with pair_count_lock:
                pair_count[0] += sum(len(n) for n in
                                     neighbors[start:end])
                if pair_count[0] > max_pairs:
                    raise TooManyMatches()
        return None

    if worker_count > 1:
        # Submit in batches of `worker_count` chunks so the pool actually
        # overlaps them. One `.result()` per submit would keep at most one
        # chunk in flight at a time, and the pass would run single-threaded
        # no matter the worker count (50k: ~162 s instead of ~17 s).
        # `should_cancel` is checked between batches — the worst-case
        # cancel latency is one batch of chunks (each
        # `_DISTANCE_CHUNK` rows).
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            for batch_start in range(0, len(starts), worker_count):
                if should_cancel is not None and should_cancel():
                    raise ScanCancelled()
                batch = starts[batch_start:batch_start + worker_count]
                futures = [pool.submit(work, start) for start in batch]
                for future in futures:
                    future.result()
    else:
        for start in starts:
            if should_cancel is not None and should_cancel():
                raise ScanCancelled()
            work(start)
    return neighbors


def seed_group(neighbors: list, order_keys: list) -> list:
    """
    The seed-based (non-transitive) groups of the neighbor lists.

    `neighbors[i]` is the (index, distance, mirrored) list of the images
    within the threshold of i (self and excluded links already removed).
    `order_keys[i]` is a comparable tuple, larger first — the caller puts
    in (degree, area, size, path) — used to order the candidate seeds.

    The image with the most neighbors seeds a group; it takes every
    not-yet-grouped image that is within its threshold; the next seed is
    chosen from the remainder, and so on. An image joins a group only if it
    meets the threshold with that group's representative, so A~B and B~C
    never force A and C into one group (plain connected components would).

    Returns a list of (seed_index, [seed_index, member indices, ...]).
    Images with no neighbors are in no group.
    """
    unassigned = {i for i in range(len(neighbors)) if neighbors[i]}
    groups = []
    for seed in sorted(unassigned, key=lambda i: order_keys[i], reverse=True):
        if seed not in unassigned:
            # Already absorbed by an earlier seed.
            continue
        members = [seed]
        for (j, _distance, _mirrored) in neighbors[seed]:
            if j in unassigned:
                members.append(j)
                unassigned.discard(j)
        groups.append((seed, members))
    return groups
