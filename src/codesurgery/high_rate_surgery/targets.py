"""Signed target preparation; one list entry means one measured observable."""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
from ..gadget_common import prepare, logical_basis_for, signature, memory_checks
from ..gf2 import matmul, nullspace, rank
from ..logical_optimization import optimize_prepared_logical
from ..symplectic import commutation_matrix, parse_pauli


def qwc(x, z):
    return all(not np.any((x[i] & z[i+1:]) ^ (z[i] & x[i+1:])) for i in range(len(x)))


def prepare_targets(code, targets=None, *, logical_paulis=None, pauli='Z',
                    logical_indices=None, logical_basis=None, optimize_support=True,
                    seed=0, optimization_trials=64):
    basis = logical_basis_for(code, logical_basis)
    if targets is not None and logical_paulis is not None:
        raise ValueError('Choose targets or logical_paulis, not both')
    if (targets is not None or logical_paulis is not None) and logical_indices is not None:
        raise ValueError('logical_indices cannot accompany explicit targets')
    if logical_paulis is not None:
        strings = [logical_paulis] if isinstance(logical_paulis, str) else list(logical_paulis)
        items = []
        for s in strings:
            a, b, sign = parse_pauli(s, code.k)
            items.append(dict(logical_x=a, logical_z=b, sign=sign))
    elif targets is not None:
        items = [targets] if isinstance(targets, (str, Mapping)) else list(targets)
    else:
        if not isinstance(pauli, str) or pauli.upper() not in ('X', 'Y', 'Z'):
            raise ValueError('pauli must be X, Y, or Z')
        indices = (list(range(code.k)) if logical_indices is None else
                   [logical_indices] if isinstance(logical_indices, (int, np.integer)) else list(logical_indices))
        items = []
        for i in indices:
            if isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or not 0 <= i < code.k:
                raise ValueError('logical_indices must be zero-based indices in [0,k)')
            a, b = np.zeros(code.k, np.uint8), np.zeros(code.k, np.uint8)
            a[i], b[i] = pauli.upper() in ('X', 'Y'), pauli.upper() in ('Z', 'Y')
            items.append(dict(logical_x=a, logical_z=b))
    if not items:
        raise ValueError('At least one nontrivial logical observable is required')
    original = [prepare(code, item, pauli=pauli, logical_basis=logical_basis)[1] for item in items]
    ox, oz = np.stack([t[0] for t in original]), np.stack([t[1] for t in original])
    if np.any(commutation_matrix(ox, oz)):
        raise ValueError('Requested logical measurements must commute pairwise')
    selected, reports = list(original), []
    # Preserve existing QWC when independent representative reduction would
    # introduce incompatible local axes. Explicit mixed inputs are never split.
    if optimize_support:
        for i, target in enumerate(original):
            proposal, report = optimize_prepared_logical(code, target, basis, method='heuristic',
                threshold=None, max_trials=optimization_trials, seed=seed+i)
            candidate = selected.copy()
            candidate[i] = proposal
            xx, zz = np.stack([t[0] for t in candidate]), np.stack([t[1] for t in candidate])
            if qwc(xx, zz) or not qwc(ox, oz):
                selected = candidate
                report['accepted_for_shared_attachment'] = True
            else:
                report['accepted_for_shared_attachment'] = False
            reports.append(report)
    x, z = np.stack([t[0] for t in selected]), np.stack([t[1] for t in selected])
    if not qwc(x, z):
        # Retry the supplied physical representatives if reduction broke their
        # compatibility. No silent logical-basis change is permitted.
        if qwc(ox, oz):
            selected, x, z = original, ox.copy(), oz.copy()
        else:
            raise ValueError('Targets are globally commuting but not qubit-wise commuting (QWC). '
                'Supply compatible representatives, or use a separately validated branching/twist-free protocol. '
                'The high-rate module will not split a product into separate measurements.')
    sig = signature(x, z, basis)
    if not np.array_equal(sig, signature(ox, oz, basis)):
        raise RuntimeError('Support reduction changed the requested logical classes')
    support = np.flatnonzero(np.any(x | z, axis=0))
    px, pz = np.zeros((len(support), code.n), np.uint8), np.zeros((len(support), code.n), np.uint8)
    for j, q in enumerate(support):
        i = np.flatnonzero((x | z)[:, q])[0]
        px[j, q], pz[j, q] = x[i, q], z[i, q]
    coordinates = (x | z)[:, support]
    sx, sz = memory_checks(code)
    f = matmul(sx, pz.T) ^ matmul(sz, px.T)
    k = nullspace(f)
    image = signature(matmul(k, px), matmul(k, pz), basis)
    # Keep every support-local stabilizer as well as the requested logical
    # space when a sparse separator for this larger allowed space exists.
    coeff = nullspace(matmul(image, nullspace(sig).T).T)
    allowed = matmul(coeff, k)
    return dict(basis=basis, source_x=sx, source_z=sz, target_x=x, target_z=z,
                target_signs=np.asarray([t[2] for t in selected], np.uint8),
                original_target_x=ox, original_target_z=oz,
                original_target_signs=np.asarray([t[2] for t in original], np.uint8),
                target_signatures=sig, target_rank=rank(sig), support=support,
                attachment_x=px, attachment_z=pz, coordinates=coordinates,
                restriction=f, allowed_kernel=allowed, optimization=reports)
