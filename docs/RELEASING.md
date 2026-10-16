# 构建与发布

从含 `pyproject.toml` 的项目根目录执行以下命令。源码采用 [MIT](../LICENSE) 许可证；
PyPI 包名、仓库地址、作者信息应由维护者按实际情况确认，本文不表示包已上传。

## 安装与用户验证

```sh
python -m pip install .
python examples/bb90_compare.py --logical ZIIIIIII IZIIIIII --scheme devised_sticking
```

科学计算测试由使用者自行运行：

```sh
python -m pip install -e ".[test]"
python -m pytest
```

可选 `.[distance]` 安装 SciPy，`.[gurobi]` 安装 Gurobi Python 接口。
Gurobi 的可用许可证需单独配置；它不属于本项目的 MIT 授权内容。

## 生成发行文件

```sh
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

`dist/` 中应得到一个 `.whl` 和一个 `.tar.gz`：

- wheel 包含 `codesurgery` 运行时源码、包元数据和许可证。
- sdist 另含文档、测试、BB90 示例及 `G.npy`、`C_sparse.npy` 两份必要输入。
- 历史计算结果、个人脚本、缓存和编辑器文件均不应进入发行文件。

`twine check` 检查发行元数据及 README 格式，不验证量子码构造或算法正确性。
每次修改源码后需要重新构建；上传时只选择当前版本对应的文件，避免混入旧版本。

## 上传

先确认 PyPI 上的发行名 `codesurgery` 可由你使用。若该名称属于别人，修改
`pyproject.toml` 中的 `project.name`；Python 导入名可以继续保持 `codesurgery`。
同一索引中的同名同版本文件不能用新文件覆盖，后续更新应提高 `project.version`。

可先向 TestPyPI 上传：

```sh
python -m twine upload --repository testpypi dist/codesurgery-0.1.0*
```

准备正式发布时，使用 PyPI 账号自己的 API token：

```sh
python -m twine upload dist/codesurgery-0.1.0*
```

上传命令会向相应平台发布文件；TestPyPI 与 PyPI 使用各自的账号与 token。
通过交互提示或密钥管理工具提供 token，不要写入源码或提交到仓库。
若更改了发行名或版本，以上文件名也须同步修改。

正式上传成功后，其他人可安装你实际发布的发行名和版本；若仍使用当前名称：

```sh
python -m pip install codesurgery
```

wheel 不包含仓库的 `examples/` 目录；示例可从源码仓库或源码压缩包获取。

参考：[Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)、
[setuptools 文件包含规则](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html)。
