# Quantum Code Surgery

**Matrix constructions and verification tools for quantum code surgery.**

CodeSurgery implements logical Pauli measurements on CSS quantum error-correcting codes using code surgery.

The package returns static check matrices and verification reports. Matrices use dense NumPy arrays with binary arithmetic over GF(2). A complete syndrome-extraction circuit, decoder, and circuit-level fault-distance proof are outside the current implementation.

[PyPI](https://pypi.org/project/codesurgery/) · [Source code](https://github.com/qiqc-error-correction/codesurgery)

## Installation

Python 3.10 or later is required. The base dependency is NumPy.

```sh
python -m pip install codesurgery
```

Optional backends and development dependencies:

```sh
python -m pip install 'codesurgery[distance]'  # SciPy/HiGHS distance and weight-reduction backend
python -m pip install 'codesurgery[gurobi]'    # Gurobi backend; requires a valid license
python -m pip install 'codesurgery[test]'      # pytest
```

To work from the source checkout:

```sh
git clone https://github.com/qiqc-error-correction/codesurgery.git
cd codesurgery
python -m pip install -e .
```

## Quick start: two BB90 logical measurements

This example separately measures logical Z on the first and second logical qubits of the IBM BB [[90,8,10]] code using devised sticking. It builds deformed codes for both the original and weight-reduced representatives.

```python
from codesurgery import compare_deformations
from codesurgery.examples import ibm_bb90_code

code = ibm_bb90_code()
comparison = compare_deformations(
    code,
    scheme="devised_sticking",
    logical_paulis=["ZIIIIIII", "IZIIIIII"],
    optimization_options={"method": "heuristic", "seed": 0},
)

print("Original weights:", comparison.summary()["before_weights"])
print("Reduced weights:", comparison.summary()["after_weights"])
print("Overhead comparison:", comparison.overhead())

original_code = comparison.original
reduced_code = comparison.optimized
comparison.save("bb90_Z1_Z2_comparison")
```

The output directory contains `original.npz` and `optimized.npz` for successful branches, together with `comparison.json`. The report records signs, physical supports, stabilizer witnesses for weight reduction, construction status, and overhead. Choose a new output directory to avoid overwriting previous results.

Each list item specifies one measurement outcome:

| Logical input | Meaning |
| --- | --- |
| `["ZIIIIIII", "IZIIIIII"]` | Measure Z1 and Z2 separately: two outcomes |
| `["ZZIIIIII"]` | Measure the product Z1 Z2: one outcome |
| `["YIIIIIII"]` | Measure Y1; use a compatible scheme such as `gauging` |
| `["XZIIIIII"]` | Measure X1 Z2 as one product observable |

The BB construction and its published parameters come from [Bravyi et al., *High-threshold and low-overhead fault-tolerant quantum memory*](https://arxiv.org/abs/2308.07915). The constructor supplies the published input distance; it does not recompute it.

## Default logical basis

If no basis is supplied, binary Gaussian elimination produces paired CSS logical generators. Write the independent check rows, after a common column permutation, as

$$
H_X^S=[A\mid I_{r_X}\mid B],\qquad
H_Z^S=[C\mid D\mid I_{r_Z}],
$$

where the three column blocks have sizes k, r_X, and r_Z, with `k = n - r_X - r_Z`. Then choose

$$
J_X^S=[I_k\mid0\mid C^T],\qquad
J_Z^S=[I_k\mid A^T\mid0].
$$

They satisfy `HZ @ JX.T = 0`, `HX @ JZ.T = 0`, and `JX @ JZ.T = I` over GF(2). The returned matrices use the original physical-qubit order. This selects a logical basis; weight reduction is a separate step.

```python
basis = code.standard_logicals()
JX, JZ = basis.jx, basis.jz
print("Logical X generators:", JX)
print("Logical Z generators:", JZ)
```

For background, see [Cleve and Gottesman](https://arxiv.org/abs/quant-ph/9607030) and the CSS block representation in [Zhang and Li, Appendix A](https://arxiv.org/html/2408.01339v3#A1).

## Your own code and logical operators

```python
from codesurgery import CSSCode, compare_deformations

code = CSSCode(hx=HX, hz=HZ, distance=d)
comparison = compare_deformations(
    code,
    scheme="devised_sticking",
    logicals=[z1, z2],  # Physical binary vectors of length n
    pauli="Z",
    optimization_options={"method": "heuristic", "seed": 0},
)
```

`HX` and `HZ` are binary matrices satisfying `HX @ HZ.T = 0 (mod 2)`. Omit `distance` if it is unknown. This argument is a trusted input-distance declaration, not a certificate computed by the constructor.

The comparison interface accepts three mutually exclusive target formats:

| Argument | Input format |
| --- | --- |
| `logical=...` | One physical Pauli string, binary vector, or `{"x": ..., "z": ..., "sign": 0}` |
| `logicals=[...]` | Multiple physical targets, each on n qubits; `pauli` specifies the type of binary vectors |
| `logical_paulis=[...]` | Pauli strings on the logical basis, each of length k |

Use `logical_basis={"jx": JX, "jz": JZ}` for a custom paired pure CSS logical basis. Its rows must commute with the corresponding stabilizers and satisfy `JX @ JZ.T = I (mod 2)`. If neither targets nor a basis are supplied, the default target is the first standard logical Z generator.

Physical strings may start with `+` or `-`; `sign=1` in a dictionary denotes a minus sign. Python indices and returned `support` arrays are zero-based. Logical Y and signed observables use the package's Pauli phase handling, rather than treating Y as an unsigned product of X and Z.

Targets must be nontrivial logical operators. The constructors validate commutation, phases, and the logical space actually read out. Multiple separate measurements must also meet the independence and compatibility conditions of the chosen scheme.

## Supported constructions and their sources

`compare_deformations(..., scheme=...)` supports six schemes. Use the native APIs when a two-branch comparison is unnecessary.

| Scheme | Native API | Main source and implemented role |
| --- | --- | --- |
| `ckbb` | `codesurgery.ckbb_deform` | [Cohen–Kim–Bartlett–Brown](https://arxiv.org/abs/2110.10794): layered generalized surgery, including supported Y/XZ constructions |
| `devised_sticking` | `codesurgery.devised_sticking.devised_deform` | [Zhang–Li](https://arxiv.org/abs/2408.01339): simultaneous sticking for independent all-X or all-Z targets, including overlapping supports |
| `explicit` | `codesurgery.explicit_gadgets.gadget_deform` | [Cross–He–Rall–Yoder](https://arxiv.org/abs/2407.18393) and [Webster–Smith–Cohen](https://arxiv.org/abs/2511.15989): target-specific explicit gadgets and the local construction in Section II.1 |
| `gauging` | `codesurgery.gauging_measurement.gauging_deform` | [Williamson–Yoder](https://arxiv.org/abs/2410.02213): auxiliary-graph gauging measurements, including general Pauli targets |
| `high_rate` | `codesurgery.high_rate_surgery.high_rate_deform` | [Zheng–Jiang–Xu](https://arxiv.org/abs/2510.08523): shared-ancilla constructions for multiple measurements |
| `branching` | `codesurgery.parallel_branching.branching_deform` | [Zhang–Li](https://arxiv.org/abs/2408.01339): branch stickers and brute-force branching; [Cowtan et al.](https://arxiv.org/abs/2503.05003): the locally compatible Pauli extension |

Adapter constructions use [Swaroop–Jochym-O'Connor–Yoder, *Universal adapters between quantum LDPC codes*](https://arxiv.org/abs/2410.03628). The native `gadget_pair_deform` and `gauging_pair_deform` APIs measure **one product** of two factors, rather than measuring the factors separately. Their support, graph, and adapter conditions are documented in the detailed notes.

Some implementation boundaries matter when choosing a scheme:

- The comparison interface accepts one complete observable per call for `ckbb`, `explicit`, and `gauging`. Write a product as one target. The top-level `deform` API still dispatches only to CKBB.
- The current devised-sticking constructor accepts all-X or all-Z independent target sets. The broader protocols in the paper do not imply that this one-sticker API accepts Y, XZ, or a mixed X/Z set.
- Multi-target branching and high-rate inputs require physical qubit-wise compatibility: on each physical qubit, two targets must have the same nonidentity Pauli or at least one identity. Global commutation alone is insufficient. Branching constructs a routing code; it does not complete a measurement.
- The explicit module implements target-specific gadgets. It does not supply the complete fixed universal architecture of Webster–Smith–Cohen. The graph-expansion heuristics are implementation choices and do not establish that architecture's universal overhead theorem.
- The high-rate module includes shared random-ancilla constructions and APIs for user-supplied HGP ancillas and connection maps. It does not automatically find arbitrary connection maps or implement a linear-PCPP generator. Its source is by **Guo Zheng**, Liang Jiang, and Qian Xu.

The `codesurgery.pauli_gates` module also provides logical H, S, and CNOT measurement plans, static surgery gadgets, and Pauli-frame updates. Relevant sources are [Zhang–Li](https://arxiv.org/abs/2408.01339), [Krishna–Poulin](https://arxiv.org/abs/1909.07424), and [Litinski–von Oppen](https://arxiv.org/abs/1709.02318). The plans assume that each merge/measurement is followed by a split back to the memory code. They are not complete circuit-level simulations or claims of optimal measurement counts; see the [Pauli-gate notes](https://github.com/qiqc-error-correction/codesurgery/blob/main/docs/PAULI_GATES.md).

## Weight reduction and overhead

The comparison interface preserves the input physical representatives in the original branch. The optimized branch searches within the same **signed stabilizer cosets**, then reconstructs and validates the deformed code.

- By default, reduction is attempted only when `wt(L) > 2d`. An unknown input distance or weight equal to `2d` does not trigger it.
- Set `optimization_options={"threshold": None}` to always attempt reduction, or supply an integer threshold.
- A heuristic keeps the original representative if it finds no improvement. An improved representative is not necessarily minimum-weight.
- Native devised sticking and single-target explicit/gauging constructors default to `optimize_logical=False`; branching/high-rate APIs enable their own reduction rules. The comparison interface controls both branches and disables a second reduction inside the constructors.

```python
overhead = comparison.overhead()
print(overhead["original"]["ancilla_qubits"])
print(overhead["optimized"]["ancilla_qubits"])
print(overhead["savings"])
```

The main metric is `ancilla_qubits = n_deformed - n_memory`, computed from the final matrices. It includes completed repetition layers, expansion, and sparsification. Reports also include total qubits, check rows, maximum check weight, and maximum qubit participation. A Y at one physical location counts once in the Pauli weight.

`savings` means original overhead minus optimized overhead; negative values indicate an increase. Check counts include redundant rows. These statistics exclude extra syndrome-extraction circuit ancillas, measurement rounds, and spacetime costs. A lighter representative need not produce a smaller deformed code or preserve distance.

Failed branches retain their status and have `None` cost fields. By default, `strict=True` raises `ComparisonConstructionError`, whose `.comparison` retains successful results and failure details. Use `strict=False` to inspect partial results directly:

```python
comparison = compare_deformations(
    code,
    logical_paulis="ZIIIIIII",
    scheme="gauging",
    strict=False,
    optimization_options={"method": "heuristic", "seed": 0},
    construction_options={"require_distance": False},
)
print(comparison.errors)
print(comparison.overhead())
```

In this example, `require_distance=False` disables the combined distance, expansion, and cycle-sparsification postprocessing. It is useful for inspecting the base construction, but its output is not a distance-preservation certificate.

## Distance, sparsification, and verification

Successful measurement constructors validate the full check commutation relations, phases, and actual logical readout. Distance requires separate evidence:

| Scheme | Default distance handling |
| --- | --- |
| CKBB | Layers default to the declared d, or n if unknown. Certification also requires support conditions; successful construction can remain uncertified. |
| Devised sticking | Repetition length defaults to d, or n if unknown. The implemented connection map has norm 1, giving a lower bound `min(d, d_R)` when d is known. |
| Explicit / gauging | `require_distance=True`: attempt certification and repair; raise if certification remains unavailable or configured limits are exceeded. |
| High-rate | `ensure_distance=True`, but the native default `strict=False` permits an uncertified result. Native `strict=True` requires certification. |
| Branching | Static distance-preservation evidence concerns the branching code. A subsequent measurement needs its own certification. |

Construction certificates can depend on the user's declared input distance. Compare the distance evidence for both branches when comparing overhead at equal protection: `.overhead()` does not enforce this condition. The comparison interface's `strict=True` is distinct from native high-rate distance strictness; pass the latter through `construction_options={"strict": True}`.

Explicit/gauging postprocessing can add graph edges for expansion, optimize the cycle basis before decongestion layers, and apply cellulation when needed. The default row-weight budget is derived from the maximum row weight of the horizontally joined original checks and memory coupling: `[HX | M_X]` for a pure-Z target, with the dual convention for pure X and a full Pauli-support count for mixed targets. Column participation has its own limit. Degree limits constrain construction cost; they alone do not prove distance or expansion. Seeded heuristic searches may fail to find an admissible construction.

After a graph change, certification is checked again on the reconstructed result. An unavailable certificate is distinguished from an exhibited low-weight logical operator. The default Gurobi limit in the explicit/gauging workflow is 60 seconds **per solver call**, not for the entire pipeline. Detailed options and assumptions are in the [distance and sparsification notes](https://github.com/qiqc-error-correction/codesurgery/blob/main/DISTANCE_PRESERVATION_ZH.md).

You can also request a static distance estimate or a threshold check directly:

```python
from codesurgery.distance_preservation import code_distance, verify_distance

report = code_distance(code, method="auto", time_limit=60.0, seed=0)
proof = verify_distance(code, 10, method="auto", time_limit=60.0, seed=0)
print(report["distance_statement"])
print(proof["target_met"])  # True: certified; False: counterexample; None: unresolved
```

Random searches supply verified upper bounds only. A timeout is not a proof of distance preservation, and MILP bounds are subject to the solver's numerical tolerances. The general stabilizer-distance interface also counts auxiliary gauge degrees of freedom as logical qubits; it cannot automatically be interpreted as a protected/dressed subsystem-distance calculation. Solver limits do not include all matrix preprocessing and reconstruction time.

## BB90 examples and command-line interfaces

The source repository retains BB90 examples. These scripts are run from the checkout after installing the package; they are not top-level scripts installed by the wheel.

```sh
python examples/bb90_compare.py --scheme devised_sticking --logical ZIIIIIII IZIIIIII
python examples/bb90_compare.py --scheme gauging --logical YIIIIIII --always
python examples/bb90_compare.py --scheme gauging --logical ZIIIIIII --distance
```

`bb90_compare.py` defaults to heuristic weight reduction. Its gauging/explicit/high-rate distance postprocessing is enabled by `--distance`; CKBB/devised repetition layers retain their native defaults. Multiple `--logical` values require a scheme that supports separate targets.

Other examples cover [gauging](https://github.com/qiqc-error-correction/codesurgery/blob/main/examples/bb90_gauging.py), [explicit gadgets](https://github.com/qiqc-error-correction/codesurgery/blob/main/examples/bb90_explicit.py), [explicit gadgets after weight reduction](https://github.com/qiqc-error-correction/codesurgery/blob/main/examples/bb90_optimized_explicit.py), [ZX adapters](https://github.com/qiqc-error-correction/codesurgery/blob/main/examples/bb90_zx_adapter.py), and a [ZX audit](https://github.com/qiqc-error-correction/codesurgery/blob/main/examples/bb90_zx_audit.py). The `bb90_explicit.py` script uses the adjacent `bb90-explicit-data/` input data. Generated output directories are not part of the distribution.

Each module has a command-line help page for its JSON inputs and construction options:

```sh
python -m codesurgery --help
python -m codesurgery.devised_sticking --help
python -m codesurgery.explicit_gadgets --help
python -m codesurgery.gauging_measurement --help
python -m codesurgery.high_rate_surgery --help
python -m codesurgery.parallel_branching --help
```

## Documentation and development

Detailed notes:

- [CKBB matrices and pseudocode](https://github.com/qiqc-error-correction/codesurgery/blob/main/ALGORITHM.md)
- [Devised sticking](https://github.com/qiqc-error-correction/codesurgery/blob/main/DEVISED_STICKING.md)
- [Explicit gadgets](https://github.com/qiqc-error-correction/codesurgery/blob/main/EXPLICIT_GADGETS.md)
- [Explicit gadgets, gauging, and adapters](https://github.com/qiqc-error-correction/codesurgery/blob/main/EXPLICIT_GADGETS_ZH.md)
- [Distance and sparsification](https://github.com/qiqc-error-correction/codesurgery/blob/main/DISTANCE_PRESERVATION_ZH.md)
- [Parallel branching](https://github.com/qiqc-error-correction/codesurgery/blob/main/docs/PARALLEL_BRANCHING.md)
- [High-rate surgery](https://github.com/qiqc-error-correction/codesurgery/blob/main/docs/HIGH_RATE_SURGERY.md)
- [Logical H, S, and CNOT through Pauli measurements](https://github.com/qiqc-error-correction/codesurgery/blob/main/docs/PAULI_GATES.md)
- [Packaging and releases](https://github.com/qiqc-error-correction/codesurgery/blob/main/docs/RELEASING.md)

After installing the `test` extra, run the test suite with `python -m pytest -q`.

## Citation

If you find **CodeSurgery** useful in your research, please consider citing our related papers below. We appreciate your acknowledgement and support of our work.

1. Guo Zhang and Ying Li, **Time-Efficient Logical Operations on Quantum Low-Density Parity Check Codes**, *Physical Review Letters* **134**, 070602 (2025). [arXiv:2408.01339](https://arxiv.org/abs/2408.01339) · [Published article](https://doi.org/10.1103/PhysRevLett.134.070602)
2. Guo Zhang, Yuanye Zhu, Xiao Yuan, and Ying Li, **Constant-Overhead Magic State Injection into qLDPC Codes with Error Independence Guarantees** (2025). [arXiv:2505.06981](https://arxiv.org/abs/2505.06981)
3. Guo Zhang, Yuanye Zhu, and Ying Li, **Accelerating Fault-Tolerant Quantum Computation with Good qLDPC Codes**, *PRX Quantum* **7**, 033012 (2026). [arXiv:2510.19442](https://arxiv.org/abs/2510.19442) · [Published article](https://doi.org/10.1103/3fj5-gn1h)

Please also cite the original method papers relevant to the constructions used in your work, listed below. The three papers above provide related research context; the package does not implement every protocol in those papers.

<details>
<summary>BibTeX for the three papers</summary>

```bibtex
@article{ZhangLi2025TimeEfficient,
  author = {Guo Zhang and Ying Li},
  title = {{Time-Efficient Logical Operations on Quantum Low-Density Parity Check Codes}},
  journal = {Physical Review Letters},
  volume = {134},
  pages = {070602},
  year = {2025},
  doi = {10.1103/PhysRevLett.134.070602},
  eprint = {2408.01339},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph},
  url = {https://arxiv.org/abs/2408.01339}
}

@misc{ZhangZhuYuanLi2025MagicStateInjection,
  author = {Guo Zhang and Yuanye Zhu and Xiao Yuan and Ying Li},
  title = {{Constant-Overhead Magic State Injection into qLDPC Codes with Error Independence Guarantees}},
  year = {2025},
  eprint = {2505.06981},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph},
  url = {https://arxiv.org/abs/2505.06981}
}

@article{ZhangZhuLi2026Accelerating,
  author = {Guo Zhang and Yuanye Zhu and Ying Li},
  title = {{Accelerating Fault-Tolerant Quantum Computation with Good qLDPC Codes}},
  journal = {PRX Quantum},
  volume = {7},
  pages = {033012},
  year = {2026},
  doi = {10.1103/3fj5-gn1h},
  eprint = {2510.19442},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph},
  url = {https://arxiv.org/abs/2510.19442}
}
```

</details>

## References for the implemented methods

- **Stabilizer standard form:** Richard Cleve and Daniel Gottesman, *Efficient Computations of Encodings for Quantum Error Correction*, Physical Review A **56**, 76–82 (1997). [arXiv:quant-ph/9607030](https://arxiv.org/abs/quant-ph/9607030). See also Daniel Gottesman, *Stabilizer Codes and Quantum Error Correction* (1997). [arXiv:quant-ph/9705052](https://arxiv.org/abs/quant-ph/9705052).
- **CKBB generalized surgery:** Lawrence Z. Cohen, Isaac H. Kim, Stephen D. Bartlett, and Benjamin J. Brown, *Low-overhead fault-tolerant quantum computing using long-range connectivity*, Science Advances **8**, eabn1717 (2022). [arXiv:2110.10794](https://arxiv.org/abs/2110.10794).
- **Devised sticking, branch stickers, and brute-force branching:** Guo Zhang and Ying Li, *Time-Efficient Logical Operations on Quantum Low-Density Parity Check Codes* (2025); full citation above. [arXiv:2408.01339](https://arxiv.org/abs/2408.01339).
- **Improved surgery and bridging codes:** Andrew W. Cross, Zhiyang He, Patrick J. Rall, and Theodore J. Yoder, *Improved QLDPC Surgery: Logical Measurements and Bridging Codes* (2024). [arXiv:2407.18393](https://arxiv.org/abs/2407.18393).
- **Explicit gadgets:** Paul Webster, Samuel C. Smith, and Lawrence Z. Cohen, *Explicit construction of low-overhead gadgets for gates on quantum LDPC codes* (2025). [arXiv:2511.15989](https://arxiv.org/abs/2511.15989).
- **Gauging measurements:** Dominic J. Williamson and Theodore J. Yoder, *Low-overhead fault-tolerant quantum computation by gauging logical operators* (2024). [arXiv:2410.02213](https://arxiv.org/abs/2410.02213).
- **Universal adapters:** Esha Swaroop, Tomas Jochym-O'Connor, and Theodore J. Yoder, *Universal adapters between quantum LDPC codes* (2024). [arXiv:2410.03628](https://arxiv.org/abs/2410.03628).
- **Branching for locally compatible Pauli representatives:** Alexander Cowtan, Zhiyang He, Dominic J. Williamson, and Theodore J. Yoder, *Parallel Logical Measurements via Quantum Code Surgery* (2025). [arXiv:2503.05003](https://arxiv.org/abs/2503.05003).
- **High-rate surgery:** Guo Zheng, Liang Jiang, and Qian Xu, *High-Rate Surgery: towards constant-overhead logical operations* (2025). [arXiv:2510.08523](https://arxiv.org/abs/2510.08523).
- **BB memory codes:** Sergey Bravyi, Andrew W. Cross, Jay M. Gambetta, Dmitri Maslov, Patrick Rall, and Theodore J. Yoder, *High-threshold and low-overhead fault-tolerant quantum memory*, Nature **627**, 778–782 (2024). [arXiv:2308.07915](https://arxiv.org/abs/2308.07915).
- **Logical gate measurement constructions:** Anirudh Krishna and David Poulin, *Fault-tolerant gates on hypergraph product codes*, Physical Review X **11**, 011023 (2021). [arXiv:1909.07424](https://arxiv.org/abs/1909.07424). Daniel Litinski and Felix von Oppen, *Lattice Surgery with a Twist: Simplifying Clifford Gates of Surface Codes*, Quantum **2**, 62 (2018). [arXiv:1709.02318](https://arxiv.org/abs/1709.02318).

## License

CodeSurgery is released under the [MIT License](https://github.com/qiqc-error-correction/codesurgery/blob/main/LICENSE). Optional dependencies, including Gurobi, have their own licenses.
