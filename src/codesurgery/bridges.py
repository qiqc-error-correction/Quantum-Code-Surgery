"""CKBB Fig. 3 connections between disjoint same-Pauli logical factors.

Each factor gets its own support-restricted Tanner graph. In particular,
an original opposite check incident to two factors has a separate auxiliary
copy in each branch. Adjacent branches are then connected by the auxiliary
qubits/checks of Fig. 3, represented by weight-two incidence rows below.
These rows are not additional two-body measurements on the memory code.

Reference: Cohen, Kim, Bartlett, Brown, Sci. Adv. 8 eabn1717 (2022),
arXiv:2110.10794, Logical CSS measurement and Fig. 3.
"""

from __future__ import annotations

import numpy as np

from .gf2 import binary_vector, in_rowspace
from .models import CSSCode


def construct_css_bridge(code, factors, pauli, layers, layer_builder, *,
                         max_entries=50_000_000):
    """Build an explicit disjoint-factor CSS parity bridge.

    ``factors`` contains at least two physical binary operator vectors, all
    of the same Pauli type. Logical factors must have disjoint physical
    supports; logical commutation alone does not satisfy this condition.
    The XOR of the factors is the requested product operator.

    This function neither repairs an individual factor's support nor assumes
    that its restricted kernel contains only that logical class. The caller
    must run the final logical-readout check on the completed construction
    and reject unwanted measurements.

    ``layer_builder`` receives ``(code, target, pauli, layers, F, B, S)`` plus
    the unconnected restriction and metadata. It constructs the alternating
    layers and validates their matrix compatibility and commutation.
    """
    if not isinstance(code, CSSCode):
        raise TypeError("code must be a CSSCode")
    if not isinstance(pauli, str) or pauli.upper() not in ("X", "Z"):
        raise ValueError("CSS bridge pauli must be X or Z")
    pauli = pauli.upper()
    if not callable(layer_builder):
        raise TypeError("layer_builder must be callable")
    try:
        inputs = list(factors)
    except TypeError as exc:
        raise ValueError("factors must contain at least two binary vectors") from exc
    if len(inputs) < 2:
        raise ValueError("A CSS parity bridge requires at least two logical factors")

    same, opposite = ((code.hx, code.hz) if pauli == "X"
                      else (code.hz, code.hx))
    vectors = []
    supports = []
    occupied = np.zeros(code.n, dtype=bool)
    for index, value in enumerate(inputs):
        vector = binary_vector(value, length=code.n,
                               name=f"factors[{index}]")
        support = np.flatnonzero(vector)
        if not len(support):
            raise ValueError(f"Logical factor {index} is the identity operator")
        if np.any((opposite @ vector) % 2):
            raise ValueError(f"Logical factor {index} does not commute with the input checks")
        if in_rowspace(vector, same):
            raise ValueError(f"Logical factor {index} is a stabilizer, not a nontrivial logical operator")
        overlap = support[occupied[support]]
        if len(overlap):
            raise ValueError(
                "CKBB Fig. 3 requires physically disjoint same-Pauli factors; "
                f"factor {index} overlaps previous factors on qubits {overlap.tolist()}"
            )
        occupied[support] = True
        vectors.append(vector)
        supports.append(support)

    target = np.bitwise_xor.reduce(np.stack(vectors), axis=0)
    if in_rowspace(target, same):
        raise ValueError("The product of the logical factors is a stabilizer, not a nontrivial logical operator")

    # Keep every branch's original-check incidence separately. Sharing an
    # original check must not collapse its two branch auxiliary qubits.
    check_indices = []
    restrictions = []
    embeddings = []
    selectors = []
    for support in supports:
        incident = np.flatnonzero(np.any(opposite[:, support], axis=1))
        restriction = opposite[np.ix_(incident, support)].copy()
        embedding = np.zeros((len(opposite), len(incident)), dtype=np.uint8)
        embedding[incident, np.arange(len(incident))] = 1
        selector = np.zeros((len(support), code.n), dtype=np.uint8)
        selector[np.arange(len(support)), support] = 1
        check_indices.append(incident)
        restrictions.append(restriction)
        embeddings.append(embedding)
        selectors.append(selector)

    widths = [len(support) for support in supports]
    row_counts = [len(restriction) for restriction in restrictions]
    width = sum(widths)
    unconnected_rows = sum(row_counts)
    bridge_count = len(vectors) - 1
    # Check the final dense allocation before materializing block-diagonal F.
    # Shared original checks can have many independent branch copies.
    if isinstance(max_entries, (bool, np.bool_)) or not isinstance(max_entries, (int, np.integer)) or max_entries < 1:
        raise ValueError("max_entries must be a positive integer")
    r = layers if layers is not None else (code.distance or code.n)
    if isinstance(r, (bool, np.bool_)) or not isinstance(r, (int, np.integer)) or r < 1:
        raise ValueError("layers must be a positive integer")
    total_rows = unconnected_rows + bridge_count
    final_n = code.n + r * total_rows + (r - 1) * width
    final_check_rows = len(same) + len(opposite) + r * width + (r - 1) * total_rows
    if 2 * final_check_rows * final_n > max_entries:
        raise ValueError("Dense bridge result exceeds max_entries; reduce layers or increase the explicit limit")
    restriction_before_bridges = np.zeros((unconnected_rows, width),
                                         dtype=np.uint8)
    support_offsets = []
    row_offset = 0
    support_offset = 0
    for restriction, branch_width in zip(restrictions, widths):
        branch_rows = len(restriction)
        restriction_before_bridges[
            row_offset:row_offset + branch_rows,
            support_offset:support_offset + branch_width,
        ] = restriction
        support_offsets.append(support_offset)
        row_offset += branch_rows
        support_offset += branch_width

    restriction = np.zeros((unconnected_rows + bridge_count, width),
                           dtype=np.uint8)
    restriction[:unconnected_rows] = restriction_before_bridges
    embedding = np.pad(np.concatenate(embeddings, axis=1),
                       ((0, 0), (0, bridge_count)))
    selector = np.concatenate(selectors, axis=0)
    bridge_pairs = []
    bridge_connections = []
    for index in range(bridge_count):
        # An internal branch's anchor participates in only two links.
        # Using a chain avoids an unbounded degree at a common endpoint.
        coordinates = (support_offsets[index], support_offsets[index + 1])
        physical = (int(supports[index][0]), int(supports[index + 1][0]))
        restriction_row = unconnected_rows + index
        restriction[restriction_row, list(coordinates)] = 1
        bridge_pairs.append(physical)
        bridge_connections.append({
            "factor_indices": (index, index + 1),
            "support_coordinates": coordinates,
            "physical_qubits": physical,
            "restriction_row": restriction_row,
        })

    # Explicitly verify the disjoint-branch pasting relation before asking
    # the shared builder to allocate the enlarged code.
    if not np.array_equal((embedding @ restriction) % 2,
                          (opposite @ selector.T) % 2):
        raise RuntimeError("Disjoint CSS bridge pasting compatibility failed")
    if np.any((restriction @ np.ones(width, dtype=np.uint8)) % 2):
        raise RuntimeError("Disjoint CSS bridge does not preserve the product target")

    metadata_extra = {
        "construction": "CKBB Fig. 3 disjoint CSS parity bridge",
        "bridge_pairs": bridge_pairs,
        "bridge_connections": bridge_connections,
        "factor_supports": supports,
        "factors": vectors,
        "factor_check_indices": check_indices,
        "factor_restrictions": restrictions,
        "factor_pasting_matrices": embeddings,
        "factor_selector_matrices": selectors,
        "factor_support_offsets": support_offsets,
        "restriction_before_bridges": restriction_before_bridges,
        "final_readout_check_required": True,
    }
    return layer_builder(
        code, target, pauli, layers, restriction, embedding, selector,
        restriction_before_bridges=restriction_before_bridges,
        metadata_extra=metadata_extra,
        max_entries=max_entries,
    )
