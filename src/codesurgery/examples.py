"""Small input codes for reproducible examples and algebraic checks."""
import numpy as np
from .models import CSSCode


def steane_code():
    h = np.array([[1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 0, 0, 1, 1, 0],
                  [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
    return CSSCode(h, h, distance=3, name="Steane [[7,1,3]]")


def four_qubit_code():
    h = np.ones((1, 4), dtype=np.uint8)
    return CSSCode(h, h, distance=2, name="[[4,2,2]]")


def ibm_bb90_code():
    """IBM [[90,8,10]] bivariate bicycle code in the authors' physical order.

    Bravyi et al., arXiv:2308.07915v2, Table 3: ell=15, m=3,
    A=x^9+y+y^2, B=I+x^2+x^7. S_r[i,(i+1) mod r]=1;
    x=S_15 tensor I_3 and y=I_15 tensor S_3. Physical columns are
    (left block, right block), with index 3*i+j inside each block.
    The distance 10 is the paper's reported static distance, not a fresh
    distance computation by this constructor.
    https://github.com/sbravyi/BivariateBicycleCodes/blob/main/decoder_setup.py
    """
    il, im = np.eye(15, dtype=np.uint8), np.eye(3, dtype=np.uint8)
    x9 = np.kron(np.roll(il, 9, axis=1), im)
    y1 = np.kron(il, np.roll(im, 1, axis=1))
    y2 = np.kron(il, np.roll(im, 2, axis=1))
    x2 = np.kron(np.roll(il, 2, axis=1), im)
    x7 = np.kron(np.roll(il, 7, axis=1), im)
    a, b = x9 ^ y1 ^ y2, np.eye(45, dtype=np.uint8) ^ x2 ^ x7
    return CSSCode(np.hstack((a, b)), np.hstack((b.T, a.T)), distance=10,
                   name="IBM BB [[90,8,10]]")


def direct_sum(*codes):
    if not codes:
        raise ValueError("Supply at least one code")
    n = sum(code.n for code in codes)
    hx = np.zeros((sum(len(code.hx) for code in codes), n), dtype=np.uint8)
    hz = np.zeros((sum(len(code.hz) for code in codes), n), dtype=np.uint8)
    col = row_x = row_z = 0
    for code in codes:
        hx[row_x:row_x + len(code.hx), col:col + code.n] = code.hx
        hz[row_z:row_z + len(code.hz), col:col + code.n] = code.hz
        col, row_x, row_z = col + code.n, row_x + len(code.hx), row_z + len(code.hz)
    distance = min(code.distance for code in codes) if all(code.distance is not None for code in codes) else None
    return CSSCode(hx, hz, distance, name="direct sum")
