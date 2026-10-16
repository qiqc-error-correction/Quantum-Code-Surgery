# Zhang–Li devised sticking：独立实现

本模块实现 [Time-Efficient Logical Operations on Quantum Low-Density Parity Checks Codes](https://arxiv.org/html/2408.01339v3) 中的 **devised sticking**：Appendix G 的 finely devised glue code，经 Appendix C/D 的 measurement sticker 得到 deformed code。入口是 `codesurgery.devised_sticking`，不调用 CKBB 的构造函数，也不修改已有 CKBB 模块、接口或命令。

## 输入与使用

```python
from codesurgery import CSSCode
from codesurgery.devised_sticking import devised_deform, build_devised_glue

code = CSSCode([[1, 1, 1, 1]], [[1, 1, 1, 1]], distance=2)

# 默认测量一个 Zhang–Li standard-form Z 逻辑生成元。
default = devised_deform(code)

# 一次同时测量两个逻辑；物理支撑可以重叠。
joint = devised_deform(code, ["ZZII", "ZIZI"])

# X 使用对偶构造；允许负号。
x = devised_deform(code, "-XXII", pauli="X")

# 等价的二进制矩阵输入；每行一个物理代表元。
joint = devised_deform(code, [[1, 1, 0, 0], [1, 0, 1, 0]])
joint.save("devised.json")
joint.save("devised.npz")
print(joint.glue.h_g, joint.glue.s, joint.glue.t)
print(joint.hx, joint.hz, joint.summary())

# 如只需要 H_G、S、T：此底层接口接受无符号二进制行矩阵。
glue = build_devised_glue(code, [[1, 1, 0, 0], [1, 0, 1, 0]])
```

`devised_deform(code, logicals=None, *, pauli="Z", logical_index=0, d_R=None, ldpc=True, max_entries=50_000_000, optimize_logical=False, logical_optimization_options=None)`：

- `logicals` 可为一个物理 Pauli 字符串、一个二进制向量、字符串列表或二进制行矩阵。字符串只接受同型纯 X 或纯 Z，类型须与 `pauli` 一致；物理顺序就是输入矩阵列顺序。
- 未提供 `logicals` 时，默认测量第 `logical_index` 个 standard-form 生成元，索引从 0 开始。不会默认同时测量所有逻辑。
- 每个目标必须是非平凡 normalizer，目标集必须在模掉原稳定子后独立。identity、稳定子、非 normalizer、空目标集、重复或逻辑相关的目标均报错。
- 接受手动输入的稳定子修饰代表元；默认保留其物理算符。设置 `optimize_logical=True` 时先乘稳定子尝试减重，再构造 glue；默认只处理重量大于 `2d` 的目标。`logical_optimization_options={"threshold": None}` 可取消这一阈值。
- `d_R` 是 repetition code 的长度；有 `d_R` 个 dual 层及 `d_R−1` 个 primal 层。默认取输入 `distance`；未给距离时取 `n`。
- `ldpc=True` 执行 bit/check duplication；`False` 保留 dressed glue，用于查看稀疏化前的矩阵。
- 这是 CSS stabilizer 输入码的单 measurement-sticker 实现。同型目标可重叠。直接输入 Y、ZX 或 X/Z 混合目标集会报错；论文的通用 Pauli 测量在 [Appendix I](https://arxiv.org/html/2408.01339v3#A9) 使用逻辑辅助块及多阶段协议，不等于一个单纯 X/Z measurement sticker。

独立命令行：

```sh
python -m codesurgery.devised_sticking input.json -o deformed.npz
python -m codesurgery.devised_sticking input.json --optimize-logical -o reduced.npz
```

这里的 `input.json` 是用户提供的输入文件。JSON 接受 `hx,hz,distance,pauli,logical/logicals,logical_index,d_R,ldpc,max_entries,optimize_logical,logical_optimization_options`；`logical` 与 `logicals` 不能同时给出。没有校验行时，用 `n` 保留矩阵宽度，例如 `{"n":2,"hx":[],"hz":[],"logical":"ZZ","d_R":1}`。原来的 `python -m codesurgery` 仍是 CKBB。

仓库保留的完整运行示例为 [BB90 减重前后构造与 overhead 比较](examples/bb90_compare.py)。

## 第一步：naked glue 和 Q=KS

以下写 Z 测量；X 测量交换 X/Z。设目标矩阵为 (J_A\in\mathbb F_2^{q\times n})，目标逻辑数为 (q)。

1. 取所有目标支撑的并集 (V)，以及与它相邻的原 X 校验行 (I)。
2. (H_N=H_X[I,V])，(S_N) 是将支撑坐标嵌入原物理坐标的选择矩阵。
3. (T_N) 把每个 naked check 映射回其原 X 校验行。因此

\[
H_XS_N^T=T_NH_N.
\]

令 (K_N) 为 (ker H_N) 的行基，仍采用用户要求的表示：

\[
Q_N=K_NS_N,\qquad M_N=Q_NJ_X^T.
\]

(k_N=\operatorname{rank}M_N) 是支撑内独立逻辑类别数。(k_N-q) 是要去掉的额外逻辑方向数；稳定子方向产生零 signature，不计入额外逻辑。

## 第二步：dressing 去掉额外逻辑方向

实现 [Appendix G Algorithm 4](https://arxiv.org/html/2408.01339v3#A7.SS2) 的线性代数构造：

- (G_0=\operatorname{rowbasis}(\ker M_N^T)K_N)，保留所有支撑内稳定子方向。
- (G_1) 是目标物理向量限制到 (V) 后的独立行基。内部允许对目标行做可逆基变换；输出 witness 仍对应每一个原输入算符及其符号。
- 补全 (G_0,G_1) 到 (ker H_N) 的基 (G_2)，并减去目标方向，使 (G_2S_N) 只有未测量逻辑分量及稳定子分量。
- 把原逻辑基变换为目标/补空间的配对基 (J_{Z,A},J_{Z,C}) 与 (J_{X,A},J_{X,C})，得到

\[
G_2S_N=UJ_{Z,C}+VH_Z,\qquad
D=(U^r)^TJ_{X,C}S_N^T,\qquad UU^r=I.
\]

程序显式验证

\[
DG_0^T=0,\qquad DG_1^T=0,\qquad DG_2^T=I.
\]

然后取

\[
H_D=\begin{bmatrix}H_N\\D\end{bmatrix},\quad
T_D=[T_N\mid0].
\]

这样 ((\ker H_D)S_N) 保留目标与支撑内稳定子，去掉其他逻辑类别。

## 第三步：将 dressed glue 稀疏化

实现 Appendix G 的 bit duplication 和 check duplication：

```text
BitDuplication(u, {a,b}):
    添加一个 bit u' 和一个 check c，c 检查 u+u'=0
    把 a,b 对 u 的边移到 u'

CheckDuplication(a, {u,v}):
    添加一个 bit w 和一个 check b
    把 a 对 u,v 的边移到 b
    a,b 都连接 w；因此 w=u+v
```

只移动 dressing 附加边，保持 naked check 行不变；新增 glue bits 的 (S) 行全零，新增 checks 的 (T) 列全零。因此始终保持

\[
H_XS^T=TH_G,\qquad
(\ker H_G)S=(\ker H_D)S_N.
\]

程序对每个原 glue bit 消减多余 dressing 边，再把 dressing check 拆为小重量校验，并检查最终行/列重量均不超过

\[
\max\{w_{\max}(H_X)+1,3\}.
\]

还使用论文末尾的 standard-form 优化：目标系数化为 ([I_q\mid P])，将 (U) 化为 ([I\mid Q])，使右逆只选择相应行。对于处于原 standard logical span 的物理代表元，检查并记录 (D) 的非零数不超过 ((k_N-q)(q+1)) 及对应大小界。任意稳定子修饰代表元仍能得到 finely devised、重量受界的 glue，但不无条件声称此优化开销界；输出实际大小。

本版明确处理两处论文记号不一致：

1. 若 (J_Z'=RJ_Z)，共轭基必须取 (J_X'=R^{-T}J_X)，才能满足 (J_X'J_Z'^T=I)。正文 Eq. (35) 等处写成了不带转置的逆；代码使用逆转置，并验证配对。
2. Algorithm 5 的 check-duplication 循环印为 (mathcal C_N)，而 Eq. (109) 计数使用 (mathcal C_D)。按 (mathcal C_N) 执行不会拆分稠密 dressing 行。代码按 Eq. (105)/(109) 和重量界处理 (mathcal C_D)，并验证 naked 行保持不变、兼容关系及重量界。

另需保留 (G_0)：fine 条件要求逻辑商空间等于目标空间，不要求整个映射 kernel 只有目标向量；其中可以包含稳定子。

## 第四步：measurement sticker 和 deformed code

设 (H_G\in\mathbb F_2^{r_G\times n_G})，(lambda=lambda_{d_R}\in\mathbb F_2^{(d_R-1)\times d_R})。Appendix C Eq. (37)/(38)：

\[
H_X^M=[I_{d_R-1}\otimes H_G\mid\lambda\otimes I_{r_G}],\qquad
H_Z^M=[\lambda^T\otimes I_{n_G}\mid I_{d_R}\otimes H_G^T].
\]

物理列顺序固定为：**原存储比特、所有 primal 层、所有 dual 层**。辅助比特数

\[
n_{\rm anc}=(d_R-1)n_G+d_Rr_G.
\]

按 Appendix D Eq. (45)/(46) pasting：原 X checks 在第一 dual 层接 (T)，第一块 sticker Z checks 在原比特上接 (S)；原 Z checks 在辅助列上全零。X 测量交换相应角色。代码返回 (\widetilde H_X,\widetilde H_Z)，并验证二者对易。

输出 `protected_x,protected_z` 按 Eq. (49)/(50) 构造并验证：未测量的同型逻辑只占原比特；其共轭逻辑延伸到 primal 层，并在最后 dual 层接 (gamma)，其中 (J_{X,C}S^T=\gamma H_G)。保护逻辑数为 (k-q)，额外 stabilizer 自由度计为辅助 gauge。

## 第五步：检查实际测量空间

glue 条件和 witness 均不足以代替最终输出检查。令最终 symplectic check 两半为 (C_X,C_Z)，其辅助部分组合为

\[
A_{\rm aux}=[C_{X,\rm aux}\mid C_{Z,\rm aux}],\qquad
L=\operatorname{rowbasis}(\ker A_{\rm aux}^T).
\]

所有辅助作用消失的 check 产品在原比特上的逻辑 signature 为

\[
\Gamma=[LC_{X,\rm data}J_Z^T\mid LC_{Z,\rm data}J_X^T].
\]

要求 (operatorname{row}(\Gamma)) **恰好等于**输入目标 signature 的行空间，其 rank 等于 (q)。额外读出、漏读目标、或 rank 相同但读出错误目标，均抛出 `DevisedReadoutError`。导出前再次验证；失败不会覆盖已有结果文件。

每个目标还提供 `measurement_witnesses` 的一行，验证其对应 check 产品等于输入的有符号物理 Pauli。原存储 checks 保持正号；校验符号与全部依赖关系一致，不产生 (-I)。

## 总体伪代码

```text
DevisedDeform(memory, targets=None, pauli=Z, d_R=None):
    未输入 targets 时，用 standard-form 的一个逻辑生成元
    检查所有 targets 是同型、非平凡、模稳定子独立的 normalizers
    V = union_support(targets)
    H_N, S_N, T_N = naked_glue(memory, V)
    K = kernel(H_N); Q = K S_N; M = Q J_conjugate^T
    G0 = kernel(M^T) K                         # 支撑内稳定子
    G1 = restricted_independent_target_basis
    G2 = complete_kernel_basis_without_active_components(G0, G1)
    将逻辑基及 U 化为 standard form
    D = right_inverse(U)^T J_conjugate,C S_N^T
    验证 D G0^T=0, D G1^T=0, D G2^T=I
    H_D = [H_N; D]
    H_G, S, T = LDPC_duplication(H_D, S_N, [T_N,0])
    验证兼容、mapped kernel、重量界和目标逻辑空间
    H_X_deformed, H_Z_deformed = paste_measurement_sticker(H_G,S,T,d_R)
    从实际 checks 消去全部辅助作用，验证 readout_space==target_space
    解每个输入算符的有符号 witness，并构造剩余 protected logical pairs
    验证对易、符号依赖、witness、protected pairs；返回独立结果对象
```

## 输出和距离范围

主要字段：`hx,hz`、`check_x,check_z,check_signs`、多个 `measurement_x,measurement_z,measurement_signs,measurement_witnesses`、`protected_x,protected_z`、`protected_k` 和 `glue.h_g,s,t`。所有测量数组均按行对应输入目标；JSON/NPZ 包含这些矩阵及中间推导元数据。

`glue.metadata` 保留 naked/dressed/glue 三阶段数据、(G_0,G_1,G_2,U,D)、稀疏化操作轨迹，以及前后的 (K,Q=KS,M)。`metadata.measurement_validation` 保留实际输出校验的辅助消元结果及逻辑 signature。

距离字段引用论文 measurement-sticker 的 **protected subsystem** 下界 (min\{d/|S|,d_R\})。本构造 (|S|=1)，所以给定有效存储距离 (d) 时记录 (min\{d,d_R\})；没有输入距离就不输出数值下界。输入距离由调用者保证，程序不计算最小距离。辅助 gauge 的存在不改变保护逻辑数；不能把所有 raw stabilizer 自由度都当成存储逻辑来解释这个下界。矩阵使用 dense NumPy，有显式资源上限；本版生成矩阵，不生成 syndrome 电路或解码器。
