"""Binary Pauli arithmetic with explicit phases; coordinates are (x | z)."""
import numpy as np


def commutation_matrix(x, z):
    return ((x @ z.T) + (z @ x.T)) % 2


def pauli_product(x, z, signs=None, selected=None):
    """Return (x,z,phase) for a product, phase in Z4 relative to P(x,z).

    P(x,z) = i**sum(x*z) X(x) Z(z), so P has literal I/X/Y/Z
    tensor factors. signs are 0 for +P and 1 for -P. A commuting
    Hermitian product has phase 0 or 2, never 1 or 3.
    """
    x, z = np.asarray(x, dtype=np.uint8), np.asarray(z, dtype=np.uint8)
    signs = np.zeros(len(x), dtype=np.uint8) if signs is None else np.asarray(signs)
    selected = range(len(x)) if selected is None else selected
    out_x, out_z = np.zeros(x.shape[1], dtype=np.uint8), np.zeros(x.shape[1], dtype=np.uint8)
    phase = 0
    for i in selected:
        next_x, next_z = out_x ^ x[i], out_z ^ z[i]
        phase = (phase + 2 * int(signs[i]) + int(np.sum(out_x * out_z))
                 + int(np.sum(x[i] * z[i])) - int(np.sum(next_x * next_z))
                 + 2 * int(np.sum(out_z * x[i]))) % 4
        out_x, out_z = next_x, next_z
    return out_x, out_z, phase


def parse_pauli(text, n):
    if not isinstance(text, str):
        raise ValueError("Pauli operator must be a string")
    text = text.upper().replace(" ", "")
    sign = int(text.startswith("-"))
    if text.startswith(("+", "-")):
        text = text[1:]
    if len(text) != n or any(p not in "IXYZ" for p in text):
        raise ValueError(f"Expected an I/X/Y/Z string of length {n}")
    return (np.array([p in "XY" for p in text], dtype=np.uint8),
            np.array([p in "ZY" for p in text], dtype=np.uint8), sign)


def pauli_string(x, z, sign=0):
    return ("-" if sign else "") + "".join("IXZY"[int(a) + 2 * int(b)] for a, b in zip(x, z))
