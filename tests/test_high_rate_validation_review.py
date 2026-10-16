"""Independent review of signed readout, physical axes, and bound ownership."""
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from codesurgery.examples import four_qubit_code, steane_code
from codesurgery.models import CSSCode
from codesurgery.high_rate_surgery import api
from codesurgery.high_rate_surgery import distance as hd
from codesurgery.high_rate_surgery.ancilla import append_intralayer


def literal_product(x, z, signs, selection):
    """Multiply literal Paulis using the single-qubit table, independently."""
    labels = "IXZY"
    table = {("X", "Y"): (1j, "Z"), ("Y", "Z"): (1j, "X"),
             ("Z", "X"): (1j, "Y"), ("Y", "X"): (-1j, "Z"),
             ("Z", "Y"): (-1j, "X"), ("X", "Z"): (-1j, "Y")}
    answer = ["I"] * x.shape[1]
    phase = 1
    for i in selection:
        phase *= (-1) ** int(signs[i])
        for j, (xx, zz) in enumerate(zip(x[i], z[i])):
            b, a = labels[int(xx) + 2 * int(zz)], answer[j]
            if a == "I":
                answer[j] = b
            elif b == "I":
                pass
            elif a == b:
                answer[j] = "I"
            else:
                factor, answer[j] = table[a, b]
                phase *= factor
    return "".join(answer), phase


def independent_group_audit(result):
    """Enumerate every final check product for these tiny fixtures."""
    n = result.original_n
    p = result.prepared
    source_group = {}
    for bits in product((0, 1), repeat=len(p["source_x"])):
        word, phase = literal_product(p["source_x"], p["source_z"],
                                      np.zeros(len(bits)), np.flatnonzero(bits))
        assert word not in source_group or source_group[word] == phase
        source_group[word] = phase
    target_signatures = {
        tuple(np.bitwise_xor.reduce(p["target_signatures"][np.flatnonzero(bits)], axis=0))
        if any(bits) else (0,) * (2 * result.code.k)
        for bits in product((0, 1), repeat=len(result.target_x))
    }
    actual_signatures = set()
    for bits in product((0, 1), repeat=len(result.check_x)):
        word, phase = literal_product(result.check_x, result.check_z,
                                      result.check_signs, np.flatnonzero(bits))
        assert phase in (-1, 1)
        if any(letter != "I" for letter in word[n:]):
            continue
        memory = word[:n]
        xx = np.array([letter in "XY" for letter in memory], np.uint8)
        zz = np.array([letter in "ZY" for letter in memory], np.uint8)
        signature = tuple(np.r_[xx @ p["basis"]["jz"].T % 2,
                                zz @ p["basis"]["jx"].T % 2])
        actual_signatures.add(signature)
        if not any(signature):
            assert source_group[memory] == phase
    assert actual_signatures == target_signatures
    for i, witness in enumerate(result.measurement_witnesses):
        word, phase = literal_product(result.check_x, result.check_z,
                                      result.check_signs, np.flatnonzero(witness))
        target, target_phase = literal_product(result.target_x, result.target_z,
                                               result.target_signs, [i])
        assert word == target + "I" * (result.n - n)
        assert phase * (-1) ** int(result.readout_sign_corrections[i]) == target_phase


@pytest.mark.parametrize("logical", ["XI", "ZI", "YI", "XX", "XZ", "YY",
                                    ["XI", "IX"], ["ZI", "IZ"], ["XI", "-XI"]])
def test_every_final_group_product_and_signed_requested_readout(logical):
    result = api.high_rate_deform(four_qubit_code(), logical_paulis=logical,
                                 ensure_distance=False, optimize_support=False, trials=2)
    independent_group_audit(result)


@pytest.mark.parametrize("optimize", [False, True])
def test_negative_stabilizer_product_and_opposite_target_signs(optimize):
    # XX * ZZ = -YY, so multiplying a positive YYXI by both stabilizers
    # produces -IIXI. The code must preserve this sign through reduction.
    code = CSSCode([[1, 1, 0, 0]], [[1, 1, 0, 0]], distance=1)
    result = api.high_rate_deform(code, targets=["YYXI", "-YYXI"],
                                 ensure_distance=False, optimize_support=optimize,
                                 max_check_weight=10, max_qubit_degree=10, trials=2)
    independent_group_audit(result)
    assert result.prepared["target_rank"] == 1
    assert result.readout_sign_corrections[0] != result.readout_sign_corrections[1]
    if optimize:
        assert result.summary()["targets"] == ["-IIXI", "IIXI"]


def test_globally_commuting_but_non_qwc_pair_is_not_silently_split():
    code = four_qubit_code()
    with pytest.raises(ValueError, match="qubit-wise"):
        api.high_rate_deform(code, targets=["XXII", "ZZII"],
                             ensure_distance=False, optimize_support=False)


def test_pure_z_report_uses_physical_axes_and_valid_witness():
    result = api.high_rate_deform(four_qubit_code(), logical_paulis="ZI",
                                 optimize_support=False, distance_method="enum", trials=1)
    report = result.metadata["distance"]
    assert result.ancilla_hadamard
    assert report["axis_convention"] == "physical output X and Z"
    witness = np.array(report["witness"], np.uint8)
    assert int(witness.sum()) == report["upper_bound"]
    if report["witness_type"] == "X":
        assert not np.any(result.hz @ witness % 2)
        assert np.any(result.subsystem["bare_z"] @ witness % 2)
    else:
        assert report["witness_type"] == "Z"
        assert not np.any(result.hx @ witness % 2)
        assert np.any(result.subsystem["bare_x"] @ witness % 2)


def direction(lower, witness):
    upper = sum(witness)
    return dict(lower_bound=lower, proven_lower_bound=lower, upper_bound=upper,
                witness=witness, witness_validated=True, distance=None,
                status="bounded", lower_bound_certificate=f"certificate_{lower}")


def report_pair(lower_x=1, lower_z=1, upper_x=3, upper_z=4, n=8):
    wx, wz = [1] * upper_x + [0] * (n-upper_x), [1] * upper_z + [0] * (n-upper_z)
    x, z = direction(lower_x, wx), direction(lower_z, wz)
    selected = x if upper_x <= upper_z else z
    return dict(x=x, z=z, lower_bound=min(lower_x, lower_z),
                proven_lower_bound=min(lower_x, lower_z), upper_bound=min(upper_x, upper_z),
                witness=selected["witness"], witness_type="X" if upper_x <= upper_z else "Z",
                witness_validated=True, low_weight_x_witnesses=[], low_weight_z_witnesses=[],
                distance=None, status="bounded")


def test_pure_z_swap_includes_top_level_witness_label():
    original = report_pair()
    result = SimpleNamespace(metadata={"growth_axis": "Z"})
    physical = api._public_distance(original, result)
    assert physical["witness_type"] == "Z"
    assert physical["witness"] == physical["z"]["witness"]
    assert original["witness_type"] == "X"


def test_merge_same_state_bounds_reselects_global_witness_and_certificate():
    old, new = report_pair(lower_x=3, upper_x=3), report_pair(upper_x=5, upper_z=4)
    merged = api._merge_bounds(old, new)
    assert merged["upper_bound"] == sum(merged["witness"]) == 3
    assert merged["witness_type"] == "X"
    assert merged["x"]["lower_bound_certificate"] == "certificate_3"
    assert old["x"]["proven_lower_bound"] == 3
    assert new["x"]["proven_lower_bound"] == 1


def test_bounds_from_before_growth_do_not_leak_to_new_matrix(monkeypatch):
    sizes = []

    def distance(hx, hz, bx, bz, **kwargs):
        n = hx.shape[1]
        sizes.append(n)
        report = report_pair(lower_x=3 if len(sizes) == 1 else 1,
                             upper_x=3, upper_z=3, n=n)
        if len(sizes) == 1:
            report["low_weight_x_witnesses"] = [[0] * (n-1) + [1]]
        return report

    def grow(state, witnesses, **kwargs):
        vector = np.ones(len(state.layers[0].internal_qubits), np.uint8)
        return append_intralayer(state, 0, vector), {"move": "review_fixture"}

    monkeypatch.setattr(api, "css_dressed_distance", distance)
    monkeypatch.setattr(api, "propose_growth", grow)
    result = api.high_rate_deform(four_qubit_code(), logical_paulis="XI",
                                 optimize_support=False, target_distance=4, max_steps=1,
                                 max_check_weight=20, max_qubit_degree=20, trials=1)
    assert sizes[1:] == [sizes[0]+1, sizes[0]+1]
    assert result.metadata["distance"]["x"]["proven_lower_bound"] == 1
    assert not result.metadata["protected_distance_certified"]


def test_distance_and_refinement_share_one_budget(monkeypatch):
    now, budgets = [0.0], []
    monkeypatch.setattr(api, "perf_counter", lambda: now[0])

    def distance(hx, hz, bx, bz, **kwargs):
        budget = kwargs["time_limit"]
        budgets.append(budget)
        now[0] += budget
        return report_pair(n=hx.shape[1], upper_x=2, upper_z=2)

    monkeypatch.setattr(api, "css_dressed_distance", distance)
    result = api.high_rate_deform(four_qubit_code(), logical_paulis="XI",
                                 optimize_support=False, max_steps=0,
                                 distance_time_limit=60, trials=1)
    assert budgets == [5, 55]
    assert result.metadata["distance_and_growth_seconds"] == 60


@pytest.mark.parametrize("pauli", ["X", "Z"])
def test_all_logicals_measured_has_no_spurious_distance_certificate(pauli):
    result = api.high_rate_deform(steane_code(), pauli=pauli,
                                 optimize_support=False, distance_method="enum", trials=1)
    assert result.protected_k == 0
    assert result.metadata["distance"]["status"] == "no_protected_logicals"
    assert result.metadata["distance"]["proven_lower_bound"] is None
    assert not result.metadata["protected_distance_certified"]
    assert not result.metadata["measurement_fault_distance_certified"]
    with pytest.raises(api.DistanceCertificationError) as exc:
        api.high_rate_deform(steane_code(), pauli=pauli, optimize_support=False,
                             distance_method="enum", trials=1, strict=True)
    assert exc.value.result.protected_k == 0


def locally_rotated_checks(code):
    x = np.vstack((code.hx, np.zeros_like(code.hz)))
    z = np.vstack((np.zeros_like(code.hx), code.hz))
    # Offline local S and H preserve the known source distance while making
    # the checks non-CSS. These operations are independent of the backend.
    z[:, 0] ^= x[:, 0]
    x[:, 1], z[:, 1] = z[:, 1].copy(), x[:, 1].copy()
    return x, z


def validate_mixed_witness(x, z, report):
    witness = report["witness"]
    vx, vz = np.array(witness["x"], np.uint8), np.array(witness["z"], np.uint8)
    assert not np.any((x @ vz + z @ vx) % 2)
    from codesurgery.gf2 import in_rowspace
    assert not in_rowspace(np.r_[vx, vz], np.hstack((x, z)))
    assert witness["weight"] == np.count_nonzero(vx | vz) == report["upper_bound"]
    assert witness["pauli"] == "".join("IXZY"[int(a)+2*int(b)] for a, b in zip(vx, vz))


@pytest.mark.parametrize("factory", [four_qubit_code, steane_code])
def test_mixed_distance_matches_locally_clifford_equivalent_css_distance(factory):
    code = factory()
    x, z = locally_rotated_checks(code)
    report = hd.symplectic_gauge_free_distance(x, z, method="enum", trials=0)
    assert report["distance"] == code.distance
    assert report["scope"] == "protected_mixed_stabilizer_gauge_free"
    assert not report["measurement_distance_certified"]
    validate_mixed_witness(x, z, report)


def test_y_counts_as_one_physical_qubit_and_paired_candidate_lists_match():
    # The sole check YY permits YI. Every nontrivial pure X/Z has weight 2.
    x = z = np.ones((1, 2), np.uint8)
    report = hd.symplectic_gauge_free_distance(x, z, method="enum", trials=0)
    assert report["distance"] == report["witness"]["weight"] == 1
    assert sum(report["witness"]["x"]) + sum(report["witness"]["z"]) == 2
    validate_mixed_witness(x, z, report)
    for wx, wz, witness in zip(report["low_weight_x_witnesses"], report["low_weight_z_witnesses"],
                                report["low_weight_pauli_witnesses"]):
        assert witness["x"] == wx and witness["z"] == wz
        assert witness["weight"] == np.count_nonzero(np.array(wx) | np.array(wz))


def test_mixed_witness_filter_uses_full_pauli_weight():
    x, z = locally_rotated_checks(four_qubit_code())
    report = hd.symplectic_gauge_free_distance(x, z, method="enum", trials=0, max_weight=1)
    assert report["proven_lower_bound"] == 2
    assert report["low_weight_x_witnesses"] == report["low_weight_z_witnesses"] == []
    assert report["witness"]["weight"] == 2


def test_mixed_incomplete_shell_does_not_certify_lower_bound():
    x, z = locally_rotated_checks(steane_code())
    report = hd.symplectic_gauge_free_distance(x, z, method="enum", trials=0, max_candidates=2)
    assert report["proven_lower_bound"] == 1
    assert report["enumerated_candidates"] == 2
    assert report["stop_reason"] == "candidate_limit"


def test_mixed_enumeration_obeys_deadline_inside_a_shell(monkeypatch):
    clock = [0.0]

    def tick():
        clock[0] += .01
        return clock[0]

    monkeypatch.setattr(hd, "perf_counter", tick)
    x, z = locally_rotated_checks(steane_code())
    report = hd.symplectic_gauge_free_distance(x, z, method="enum", trials=0, time_limit=.13)
    assert report["proven_lower_bound"] == 1
    assert report["enumerated_candidates"] < 3 * x.shape[1]
    assert report["stop_reason"] == "time_limit"


def test_mixed_heuristic_only_has_no_unearned_lower_bound():
    x, z = locally_rotated_checks(steane_code())
    a = hd.symplectic_gauge_free_distance(x, z, method="heuristic", trials=4, seed=2)
    b = hd.symplectic_gauge_free_distance(x, z, method="heuristic", trials=4, seed=2)
    assert a["proven_lower_bound"] == 1 and a["distance"] is None
    assert a["witness"] == b["witness"]
    validate_mixed_witness(x, z, a)


@pytest.mark.parametrize("method", ["gurobi", "auto"])
def test_mixed_missing_gurobi_is_reported_without_raising(monkeypatch, method):
    monkeypatch.setattr(hd, "_solve_gurobi", lambda *a, **kw: {"status": "unavailable"})
    x, z = locally_rotated_checks(four_qubit_code())
    report = hd.symplectic_gauge_free_distance(x, z, method=method, trials=0)
    assert report["solver"]["status"] == "unavailable"
    assert report["proven_lower_bound"] == (2 if method == "auto" else 1)
    validate_mixed_witness(x, z, report)


@pytest.mark.parametrize("status,bound,expected", [(9, 2, 2), (12, 2, 1), (9, 100, 1)])
def test_mixed_gurobi_lower_bound_validation(monkeypatch, status, bound, expected):
    monkeypatch.setattr(hd, "_solve_gurobi", lambda *a, **kw: {
        "status": status, "objective_bound": bound, "candidate": None, "solution_count": 0})
    x, z = locally_rotated_checks(steane_code())
    report = hd.symplectic_gauge_free_distance(x, z, method="gurobi", trials=0)
    assert report["proven_lower_bound"] == expected


def test_mixed_invalid_gurobi_witness_is_not_a_certificate(monkeypatch):
    x, z = locally_rotated_checks(four_qubit_code())
    monkeypatch.setattr(hd, "_solve_gurobi", lambda *a, **kw: {
        "status": 2, "candidate": [0] * 8, "objective": 0, "solution_count": 1})
    report = hd.symplectic_gauge_free_distance(x, z, method="gurobi", trials=0)
    assert report["proven_lower_bound"] == 1
    assert report["distance"] is None
    validate_mixed_witness(x, z, report)


def test_mixed_stabilizer_state_is_not_measurement_distance_certificate():
    x, z = np.array([[1]], np.uint8), np.array([[1]], np.uint8)
    report = hd.symplectic_gauge_free_distance(x, z, method="enum")
    assert report["status"] == "no_protected_logicals"
    assert report["proven_lower_bound"] is report["upper_bound"] is report["distance"] is None
    assert not report["measurement_distance_certified"]


def test_mixed_distance_rejects_noncommuting_checks():
    with pytest.raises(ValueError, match="commuting"):
        hd.symplectic_gauge_free_distance([[1], [0]], [[0], [1]])


def test_general_non_css_five_qubit_code_matches_full_pauli_enumeration():
    from codesurgery.gf2 import in_rowspace
    words = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    x = np.array([[letter in "XY" for letter in row] for row in words], np.uint8)
    z = np.array([[letter in "YZ" for letter in row] for row in words], np.uint8)
    minimum = 6
    stabilizers = np.hstack((x, z))
    for assignment in product(range(4), repeat=5):
        xx = np.array([value & 1 for value in assignment], np.uint8)
        zz = np.array([bool(value & 2) for value in assignment], np.uint8)
        if np.any((x @ zz + z @ xx) % 2) or in_rowspace(np.r_[xx, zz], stabilizers):
            continue
        minimum = min(minimum, int(np.count_nonzero(xx | zz)))
    report = hd.symplectic_gauge_free_distance(x, z, method="enum", trials=0)
    assert report["distance"] == minimum == 3
    validate_mixed_witness(x, z, report)


@pytest.mark.parametrize("logical", ["YI", "XZ", "YY"])
def test_api_certifies_gauge_free_mixed_distance(logical):
    result = api.high_rate_deform(four_qubit_code(), logical_paulis=logical,
                                 optimize_support=False, distance_method="enum", trials=1)
    report = result.metadata["distance"]
    assert result.gauge_qubits == 0
    assert report["scope"] == "protected_mixed_stabilizer_gauge_free"
    assert report["proven_lower_bound"] >= 2
    assert result.metadata["protected_distance_certified"]
    assert not result.metadata["measurement_fault_distance_certified"]
    validate_mixed_witness(result.check_x, result.check_z, report)


def test_api_all_y_logicals_measured_keeps_readout_distance_unverified():
    result = api.high_rate_deform(steane_code(), pauli="Y", optimize_support=False,
                                 distance_method="enum", trials=1)
    assert result.protected_k == 0
    assert result.metadata["distance"]["status"] == "no_protected_logicals"
    assert not result.metadata["protected_distance_certified"]
    assert not result.metadata["measurement_fault_distance_certified"]
