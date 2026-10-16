"""An infeasible starting representative can become feasible after reduction."""
import json

import numpy as np
import pytest

from codesurgery.models import CSSCode
from codesurgery.parallel_branching import branching_deform
from codesurgery.parallel_branching.models import jsonable
from codesurgery.parallel_branching.policy import BranchingBudgetError


def reducible_code():
    return CSSCode([[1, 1, 0]], [[1, 1, 0]], distance=1)


def test_support_reduction_rescues_initially_infeasible_zero_row_budget():
    result = branching_deform(reducible_code(), targets=["ZZZ"],
        max_check_increase=0, logical_optimization_options={"threshold": None})
    report = result.metadata["support_optimization"]
    assert report["before_weights"] == [3]
    assert report["after_weights"] == [1]
    assert report["before_ancillas"] is None
    assert report["candidate_ancillas"] == 0
    assert report["after_ancillas"] == 0
    assert report["accepted"] == [True]
    assert result.routing["projected_original_row_growth"] == [0, 0]
    assert result.routing["projected_original_column_growth"] == [0, 0, 1]
    np.testing.assert_array_equal(result.source["prepared_z"], [[0, 0, 1]])
    assert result.validate()["measurement_attachment_limits_satisfied"]
    # Absence of a feasible incumbent is serialized as null, never Infinity.
    encoded = json.dumps(jsonable(report), allow_nan=False)
    assert '"before_ancillas": null' in encoded


def test_negative_signed_representative_keeps_its_sign_when_rescued():
    result = branching_deform(reducible_code(), targets=["-ZZZ"],
        max_check_increase=0, logical_optimization_options={"threshold": None})
    assert result.leaf_signs.tolist() == [1]
    assert result.summary()["leaf_paulis"] == ["-IIZ"]
    assert result.validate()["signed_relations_valid"]


def test_disabled_support_reduction_reports_current_representative_failure():
    with pytest.raises(BranchingBudgetError, match="current prepared representatives.*disabled"):
        branching_deform(reducible_code(), targets=["ZZZ"],
                         max_check_increase=0, optimize_logical=False)


def test_no_improvement_still_fails_after_search_without_logical_class_claim():
    code = CSSCode([[1, 1, 0]], [], distance=1)
    with pytest.raises(BranchingBudgetError, match="not an impossibility proof"):
        branching_deform(code, targets=["ZZI"], max_check_increase=0,
                         logical_optimization_options={"threshold": None})


def test_zero_column_budget_remains_infeasible_after_support_reduction():
    with pytest.raises(BranchingBudgetError, match="current prepared representatives"):
        branching_deform(reducible_code(), targets=["ZZZ"], max_qubit_increase=0,
                         logical_optimization_options={"threshold": None})


@pytest.mark.parametrize("kwargs", [dict(max_check_increase=-1),
    dict(max_qubit_increase=True), dict(branching_mode="invalid")])
def test_invalid_policy_options_still_propagate_as_configuration_errors(kwargs):
    with pytest.raises(ValueError) as exc:
        branching_deform(reducible_code(), targets=["ZZZ"], **kwargs)
    assert not isinstance(exc.value, BranchingBudgetError)


def test_unknown_distance_threshold_skip_does_not_fabricate_a_feasible_route():
    code = CSSCode([[1, 1, 0]], [[1, 1, 0]])
    with pytest.raises(BranchingBudgetError, match="configured support-optimization attempt"):
        branching_deform(code, targets=["ZZZ"], max_check_increase=0)

