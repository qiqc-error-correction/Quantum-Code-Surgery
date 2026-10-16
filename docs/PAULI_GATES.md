# Logical H, S, and CNOT through Pauli measurements

`codesurgery.pauli_gates` decomposes logical gates into ordered Pauli measurements, calls a surgery backend for each distinct observable, and resolves the Pauli corrections from supplied outcomes. The default backend is `gauging`.

The first **logical qubit of the input code** is reserved as a reusable ancilla by default (`ancilla=0`). Other logical indices keep their original labels. Preparing and measuring this ancilla discards its previous state, so it must not hold unknown data or also serve as a gate operand.

## Measurement sequences

Let a denote the logical ancilla, d a data qubit, and c and t the CNOT control and target. Each Pauli product is one joint measurement. Outcome bit 0 means eigenvalue +1; bit 1 means -1. The b_i below are **effective outcomes**, after accounting for the current Pauli frame.

| Gate | Ancilla preparation | Ordered measurements | Data correction, up to global phase |
| --- | --- | --- | --- |
| `H(d)` | `Z_a=+1` | `X_d X_a → b1`; `Z_d X_a → b2`; `Z_a → b3` | `X_d^(b1 XOR b2) Z_d^(1 XOR b1 XOR b2 XOR b3)` |
| `S(d)` | `X_a=+1` | `Z_d Y_a → b1`; `Z_a → b2` | `Z_d^(b1 XOR b2)` |
| `CNOT(c,t)` | `X_a=+1` | `Z_c Z_a → b1`; `X_a X_t → b2`; `Z_a → b3` | `Z_c^b2 X_t^(b1 XOR b3)` |

Here `S=diag(1,i)`. The H sequence keeps the data in its original logical slot. Its Z correction has a constant term, so all-zero outcomes do not imply an identity correction. After the final ancilla Z readout, record or apply `X_a` raised to that effective outcome to reset the ancilla to the +Z state.

Preparation is itself a measurement: a Z outcome p is followed by `X_a^p` to prepare |0>, or an X outcome p by `Z_a^p` to prepare |+>. Including preparation, H and CNOT each use four measurements, and S uses three. The compiler does not generate outcomes when none are supplied.

[Zhang and Li, Appendix I.3](https://arxiv.org/html/2408.01339v3#A9.SS3) and [Krishna and Poulin, Lemma 21](https://arxiv.org/html/1909.07424#S5.SS2) discuss logical Clifford gates through Pauli measurements. The table specifies this package's concrete ideal measurement identities: its H keeps the original data slot, and its S uses a directly measurable Y observable. These sequences are not presented as exact reproductions of the Zhang–Li figures or as optimal measurement counts. The CNOT parity-measurement sequence is also described by [Litinski and von Oppen, Figure 3](https://arxiv.org/html/1709.02318#S3).

## Python API

```python
from codesurgery.examples import ibm_bb90_code
from codesurgery.pauli_gates import compile_gates

plan = compile_gates(
    ibm_bb90_code(),
    [("H", 1), ("S", 1), ("CNOT", 1, 2)],
    ancilla=0,
    backend="gauging",
    # Retain the backend's default distance processing.
)
print(plan.summary())
plan.save("pauli-gates-output", matrix_format="npz")
```

To inspect the decomposition without constructing matrices or computing distances:

```python
plan = compile_gates(code, [("H", 1), ("S", 1)], build=False)
for step in plan.measurements:
    print(step.id, step.role, step.logical_pauli)
```

Single-gate calls are also available:

```python
from codesurgery.pauli_gates import compile_gate

h_plan = compile_gate(code, "H", target=1)
cnot_plan = compile_gate(code, "CNOT", control=1, target=2)
```

A custom `logical_basis={"jx": JX, "jz": JZ}` may be supplied. Otherwise, binary Gaussian elimination selects paired CSS logical generators. The entire plan uses the same basis and labels. Logical Y uses `i Xbar Zbar`, including the required physical Pauli sign.

For a small-code inspection without distance postprocessing:

```python
from codesurgery.examples import direct_sum, four_qubit_code

code = direct_sum(four_qubit_code(), four_qubit_code())  # [[8,4,2]]
plan = compile_gates(
    code,
    [("H", 1), ("S", 1), ("CNOT", 1, 2)],
    backend_options={"require_distance": False},
)
```

This setting does not certify distance. The compiler retains the requested observables and backend; it does not split products, change backends, or disable protection to make a failed construction succeed.

## Outcomes and the Pauli frame

Each measurement has an ID such as `g0_prep` or `g0_m1`. Preparation outcomes are required. Decode physical records with the gadget's signed readout witness and account for split-induced Pauli corrections. Pass the resulting outcomes of the **requested logical Pauli** to `resolve`.

```python
# measured_bits maps measurement IDs to actual or explicitly simulated 0/1 outcomes.
resolution = plan.resolve(measured_bits, outcome_format="bits")
print(resolution["correction_pauli"])
print(resolution["data_correction_pauli"])
print(resolution["ancilla_correction_pauli"])
```

A complete list in `plan.measurements` order is also accepted. Use `outcome_format="eigenvalues"` for +1/-1 inputs. The result includes raw and effective outcomes, outcome flips, and the final frame. `correction_pauli` is a string on all k original logical positions.

By default, `resolve` records preparation corrections, gate corrections, and ancilla resets in the frame. For a current frame P, requested measurement M, and raw bit r:

```text
effective_bit = r XOR symplectic_product(P, M)
```

Conditional frame updates are evaluated using effective bits in execution order. This already accounts for propagation through the measurement sequence; no extra gate-boundary conjugation is needed. The same symplectic rule covers Y. If corrections are physically applied during execution, consume or clear the corresponding frame at that boundary to avoid counting them twice.

## Backends

| Backend | Construction | Conditions |
| --- | --- | --- |
| `gauging` (default) | Gauging measurement | General Pauli targets; retains native distance processing |
| `ckbb` | CKBB | Retains the support and layer conditions |
| `explicit` | Explicit gadgets | May reject restrictions or extra logical readout; products are not automatically split |
| `devised_sticking` | Devised sticking | Current API accepts pure X or pure Z; mixed targets in H/S raise an error |
| `high_rate` | High-rate surgery | Retains its search budgets, distance options, and native `strict` semantics |

`backend_options` is forwarded to each distinct observable's construction, with code, basis, and targets protected from overrides. Equal observables can share a matrix construction, while their measurement steps and outcomes remain distinct. A construction failure identifies the backend and target.

Distance limits apply to backend solver calls, not the whole multi-gate plan. Constructing several distinct gadgets may therefore take longer than one solver's time limit. Distance postprocessing is disabled only by an explicit option.

## Pseudocode

```text
COMPILE_GATES(code, gates, ancilla=0, basis=None, backend="gauging"):
    choose the supplied basis or paired CSS generators from Gaussian elimination
    validate ancilla and gate operands; reserve ancilla exclusively
    for each gate:
        generate ordered preparation, joint-measurement, and ancilla-readout steps
        assign each step its own ID and conditional frame updates
        for each complete logical Pauli L in these steps:
            if matrix construction is enabled and L is not cached:
                obtain the signed physical representative of L in the fixed basis
                construct the selected surgery gadget on the original memory code
                validate check commutation, phases, and exactly the readout span(L)
                retain matrices, readout witnesses, and distance status
    return the plan, distinct gadgets, frame rules, and backend statuses

RESOLVE(plan, raw_outcomes, initial_frame=I):
    process measurements in execution order
    obtain each effective outcome using the current frame
    update the frame using the gate's conditional corrections
    return the final correction and step-by-step trace
```

Every step assumes a complete **merge, nondestructive measurement, and split back to the original memory code**. The next step uses that memory code and the same logical basis. The ordered steps must not be combined into one simultaneous check matrix: their noncommuting measurements are intentional.

## Command line and export

Run from the source checkout, or omit `PYTHONPATH=src` after installation:

```sh
PYTHONPATH=src python -m codesurgery.pauli_gates \
  --example eight --gate H:1 --gate S:1 --gate CNOT:1:2 \
  --backend gauging --no-ensure-distance -o pauli-gates-output
```

To export only a BB90 logical plan:

```sh
PYTHONPATH=src python -m codesurgery.pauli_gates \
  --example bb90 --gate H:1 --gate S:1 --gate CNOT:1:2 \
  --decompose-only -o bb90-gate-plan
```

Input JSON accepts `example` or `hx/hz`, together with `distance`, `logical_basis`, `gates`, `ancilla`, `backend`, `backend_options`, and `build`. For example:

```json
{
  "example": "eight",
  "gates": [["H", 1], ["S", 1], ["CNOT", 1, 2]],
  "ancilla": 0,
  "backend": "gauging",
  "backend_options": {"require_distance": false}
}
```

`--backend-options '{"require_distance": false}'` overrides corresponding backend options; `@options.json` reads them from a file. `--outcomes @outcomes.json` also resolves corrections, and `--outcome-format eigenvalues` selects the eigenvalue convention. Explicit command-line gates replace the JSON gate list. Existing exports require `--overwrite` to be replaced.

Exports contain `plan.json` and per-step matrix files (`.npz` or `.json`), with a step-to-file map in the plan. Equal observables share an in-memory construction but are exported under step-specific filenames. Supplied outcomes additionally produce `corrections.json`. For single-observable surgery examples, see [BB90](../examples/bb90_compare.py).

## What is certified

The gate formulas describe ideal logical measurement instruments, including their conditional corrections. Every constructed gadget also validates its static check algebra and signed logical readout. Static distance evidence comes from the selected backend.

Physical execution must supply nondestructive measurement, splitting back to the memory code, and decoding. This module does not construct complete syndrome-extraction circuits or certify the spacetime fault distance of preparation, splitting, record errors, and feedforward. Its `measurement_circuit_distance_certified` field remains `False`; a static distance certificate for one gadget is not a circuit-level certificate for the gate sequence.
