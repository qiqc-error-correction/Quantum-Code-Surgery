# Building and releasing

Run these commands from the project root containing `pyproject.toml`. CodeSurgery uses the [MIT License](../LICENSE). Publishing a source commit to GitHub and uploading a release to PyPI are separate operations.

## Installation and local verification

```sh
python -m pip install .
python examples/bb90_compare.py --logical ZIIIIIII IZIIIIII --scheme devised_sticking
```

To install development dependencies and run the tests yourself:

```sh
python -m pip install -e '.[test]'
python -m pytest
```

The optional `.[distance]` extra installs SciPy; `.[gurobi]` installs Gurobi's Python interface. A Gurobi license must be configured separately and is not covered by this project's MIT license.

## Build distribution files

Before an update, increment `project.version` in `pyproject.toml` to a version that has not been uploaded to the target index. PyPI does not allow replacing previously uploaded distribution files. README changes on PyPI also require a new release.

```sh
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

A build produces a wheel (`.whl`) and a source archive (`.tar.gz`). The wheel contains runtime Python modules, metadata, and the license. The source archive additionally includes documentation, tests, BB90 examples, and the required `G.npy` and `C_sparse.npy` example inputs. Generated results, caches, personal scripts, and editor files should be excluded by the distribution rules.

`twine check` checks metadata and README rendering; it does not validate quantum-code constructions. Rebuild after changing the source and inspect only the intended version's files. Old versions may remain in `dist/`, so upload explicit filenames.

## Upload

The following commands illustrate an update to **0.1.1**. Use them only after setting that version and building the matching files; substitute your chosen version if different.

First upload to TestPyPI if a trial release is desired:

```sh
python -m twine upload --repository testpypi --username __token__ \
  dist/codesurgery-0.1.1-py3-none-any.whl dist/codesurgery-0.1.1.tar.gz
```

For the production release:

```sh
python -m twine upload --username __token__ \
  dist/codesurgery-0.1.1-py3-none-any.whl dist/codesurgery-0.1.1.tar.gz
```

TestPyPI and PyPI have separate accounts and API tokens. Supply the token for the selected index through the interactive prompt or a supported credential manager. Do not put tokens in source files or commit them to Git.

After a production release, users can install the published package:

```sh
python -m pip install --upgrade codesurgery
```

The wheel does not contain the repository's top-level `examples/` directory. Obtain those scripts from the [source repository](https://github.com/qiqc-error-correction/codesurgery) or source archive.

References: [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/) and [setuptools distribution-file rules](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html).
