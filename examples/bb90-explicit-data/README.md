# BB90 explicit gadget inputs

`G.npy` and `C_sparse.npy` are inputs to `../bb90_explicit.py` and
`tests/test_bb90_explicit.py`. They are retained in the source distribution.

- `G.npy`: an instance-specific Hamilton cycle for the standard-form BB90 Z₁
  representative, with shape 24 × 24.
- `C_sparse.npy`: a sparse cycle-check basis for the augmented restriction [F; G],
  with shape 35 × 58.

These arrays depend on that exact physical representative and column ordering.
They are not a prescription for other codes or targets. The example recomputes
the algebraic and expansion checks; no saved certificate is trusted.

Generated verification reports and deformed-code matrices are not distributed.
Running the example writes fresh output to `results/bb90-explicit/` by default.
