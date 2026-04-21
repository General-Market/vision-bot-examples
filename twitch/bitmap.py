from web3 import Web3

MAX_BITMAP_BYTES = 1024
MAX_BITMAP_BITS = MAX_BITMAP_BYTES * 8


def encode_bitmap(bets: list[str], count: int) -> bytes:
    if len(bets) < count:
        raise ValueError(f"Bitmap underflow: {len(bets)} bets for {count} markets")
    if count > MAX_BITMAP_BITS:
        raise ValueError(f"Bitmap overflow: {count} > {MAX_BITMAP_BITS}")
    bitmap = bytearray(MAX_BITMAP_BYTES)
    for i in range(count):
        if bets[i] == "UP":
            bitmap[i // 8] |= 1 << (7 - (i % 8))
    return bytes(bitmap)


def hash_bitmap(bitmap: bytes) -> bytes:
    return Web3.keccak(bitmap)
