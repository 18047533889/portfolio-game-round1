# Round 1 最终交付报告

日期：2026-09-19

---

## 一、交付物

**只提交一个文件：`submission/portfolio_round1.py`**

- SHA256：见 `reports/verification.json` 的 `standalone_sha256`
- 行数 527，23.1 KB，无外部资源依赖
- 内部唯一真源是 `src/portfolio_game/core.py`，由 `tools/build_submission.py` 原文嵌入，两者一致性由测试保证

---

## 二、作业要求逐条核对

| # | 要求（英文原文要点） | 本次是否满足 | 证据 |
|---|---|---|---|
| 1 | 提交一个 `.py` 文件 | 是 | 仅 `submission/portfolio_round1.py` |
| 2 | 类名精确为 `CVXPYPortfolio`（区分大小写） | 是 | 第 502 行 `class CVXPYPortfolio(BaseOptimization)` |
| 3 | 继承 `skfolio.optimization.BaseOptimization` | 是 | 同上；`tests/test_skfolio_integration.py` 断言 `issubclass` |
| 4 | `fit(self, X, y=None)` 且先调用 `validate_data(self, X)` | 是 | 第 510 行；NaN 输入时才退到 `ensure_all_finite=False` |
| 5 | `self.weights_` 为 `(n_assets,)` 的 numpy 数组且全部有限 | 是 | `check_weights` 强制校验 |
| 6 | `shortselling = FALSE`，权重非负 | 是 | `project_simplex` + `_normalise` 显式拒绝实质负权重 |
| 7 | `leverage = 1`，`sum(weights) == 1` | 是 | `_normalise` 把残差补到最大持仓上，容差 1e-10 |
| 8 | 不允许使用等权组合 | 是 | `_finish` 逐候选检查 `is_equal_weight`，命中即跳过 |
| 9 | 不重写 `__init__` | 是 | 全文无 `def __init__`；`tests/test_build.py` 用 AST 断言 |
| 10 | 文件内不含回测代码、不下载数据、不读写文件、不联网 | 是 | 全部导入仅 stdlib + numpy/scipy/sklearn/skfolio |
| 11 | 只用 `X` 的信息，不依赖上次 `fit` 的状态 | 是 | `allocate()` 是纯函数，测试断言重复调用结果逐位相同 |
| 12 | 能稳健处理不同数据集 | 是 | 见第四节；1 资产 ~ 1455 资产、1 行 ~ 8312 行均通过 |

评分口径（`lookback=252`、`optimize_every=20`、Sharpe 10% / max drawdown 10% / annual return 10% / **failure rate 70%**）已在 `docs/round1-requirements-zh.md` 中逐句翻译。

---

## 三、算法（最终版）

```
252 期历史收益
  → Ledoit–Wolf 收缩 → 长期相关矩阵 R
  → Marchenko–Pastur 特征值去噪（新增）
  → 指数加权近期波动 D_h
  → C = (1-ρ)·C_slow + ρ·D_h R D_h ，ρ = 0.25
  → HRP 树结构锚 b
  → 凸二次规划：min  w'Cw/(b'Cb) + λ·||w-b||²/||b||²
                 s.t. w ≥ 0, 1'w = 1
     λ = 0.25（原为 1.0）
  → 单纯形投影梯度 + 一阶对偶间隙收敛证书
```

### 为什么加去噪

相关矩阵的样本估计在 N 接近 T 时噪声极大。MP 律给出了「信号与噪声分界」的理论位置 λ₊ = (1+1/√q)²，q = T/N。低于 λ₊ 的特征方向在统计上与噪声不可区分，把它们合并到共同均值可以在**不改变估计目标**的前提下降低估计方差。

实测（`sp500`、`ftse100`、`factors` 三个数据集全期）加入去噪后最大回撤三项全部改善：0.4190→0.4119、0.4217→0.4038、0.3898→0.3881。

### 为什么锚惩罚从 1.0 降到 0.25

锚 `b` 的作用是**防止小资产池上的集中度失控**，这是它不可替代的价值：

| 配置 | `factors`（5 资产）最大权重 | 有效持仓数 1/Σw² |
|---|---|---|
| 纯最小方差（无锚） | 0.833 | 1.40 |
| λ = 0.25 | 0.535 | 2.59 |
| λ = 1.00 | 0.490 | 2.98 |

而 λ = 1.0 的代价是让 HRP 这个只看树结构、不看风险数值的启发式主导了解，最大回撤因此明显变差。λ = 0.25 是在「保留锚的分散化保险」与「让协方差信息说话」之间的折中。

---

## 四、实测结果

### 真实市场数据（`optimize_every = 20`，252 天回看）

| 数据集 | 资产数 | Sharpe | Max Drawdown | 年化收益 | 失败窗口 |
|---|---|---|---|---|---|
| sp500 | 20 | 0.9494 | 0.4845 | 0.1442 | **0 / 403** |
| ftse100 | 64 | 1.0228 | 0.4166 | 0.1369 | **0 / 285** |
| factors | 5 | 0.6503 | 0.3935 | 0.1086 | **0 / 100** |
| sp500_index | 1 | 0.5072 | 0.7362 | 0.0932 | **0 / 403** |
| nasdaq | 1455 | 1.1892 | 0.3916 | 0.2798 | **0 / 55** |

升级前后对比（同口径）：

| 数据集 | 指标 | 升级前 | 升级后 |
|---|---|---|---|
| sp500 | Sharpe | 0.9820 | 0.9494 |
| sp500 | **MaxDD** | 0.5340 | **0.4845** |
| ftse100 | Sharpe | 0.9917 | **1.0228** |
| ftse100 | **MaxDD** | 0.4304 | **0.4166** |
| factors | Sharpe | 0.6375 | **0.6503** |
| factors | **MaxDD** | 0.4001 | **0.3935** |

**最大回撤在 3/3 个数据集上改善**；夏普与年化收益互有升降，差异在噪声量级。选型脚本见 `outputs/round1-audit/bench_ship.py`，证据见 `docs/method-selection-zh.md`。

### 为什么把稠密风险估计上限从 256 提到 2048

上限原本是 256，超过就退为逐资产反波动率加权。这道护栏的代价在最宽的数据集上是可测的：

| nasdaq（1455 资产，55 折） | Sharpe | MaxDD | 年化收益 | 失败折 |
|---|---|---|---|---|
| 上限 256（旧） | 0.6036 | 0.4746 | 0.1494 | 0 / 55 |
| 上限 2048（现） | **1.1892** | **0.3916** | **0.2798** | 0 / 55 |

也就是说旧的上限让这个数据集完全用不上相关结构：它的权重与纯反波动率逐位相同。放开后走的是协方差链路（去噪相关 + 单链接树 + 逆方差簇风险），55 折状态全部是 `fallback_hrp`，无一次失败。代价是单次 fit 从 0.01 秒升到 2.8 秒。

阈值用实测定，不由手感定。单次 fit 在 256 资产约 0.02 秒、1455 资产约 2.8 秒、2048 资产约 6.4 秒、2900 资产约 12.3 秒（树步骤是 O(n³)，所以这是成本曲线而非统计曲线）。取 2048 的理由是它覆盖所有标准宽横截面并留出余量，同时把最坏一折压在数秒级；超过 2048 才退回 O(TN) 的逐资产配置。回归测试 `test_wide_cross_section_uses_the_covariance_path` 会在这条上限被改小到覆盖不住标准横截面时失败。

### 时间切分稳健性（3 数据集 × 3 时间段 = 9 个评价块）

把每个数据集的回测收益序列按 2018-01-01 切开后分别算指标，再对候选池做排名百分位：

- 最大回撤在 **8/9** 个评价块中改善
- 综合稳健得分 0.1537（升级前 0.1414）

这也是没有选择 λ = 0.10 的原因：它在全期得分更高（0.1660），但 5 资产组合的最大权重会升到 0.602，在小资产池上引入不必要的尾部风险——而失败率占 70% 权重，稳健优先。

### 鲁棒性验证

| 项目 | 结果 |
|---|---|
| 老师原版 `self_test.py` | `Basic checks passed`，127 个组合，**0 失败 / 0 降级**，年化 Sharpe 0.92 |
| `pytest` | 95 项：94 通过 / 1 跳过 / **0 失败** |
| 合成边界压力测试（240 例） | **最终失败 0**；89 例走了显式记录的降级路径 |
| `tools/verify_repo.py --require-integration` | 退出码 0，`real_skfolio_preflight: PASSED`，`ready_for_teacher_submission: true` |
| 单次 fit 耗时 | 压力用例 p95 = 0.040 秒；20~64 资产真实横截面约 0.003~0.008 秒；1455 资产宽横截面约 2.8 秒 |

240 例压力测试覆盖：1/2/5/20/63/252/400 行 × 2/3/5/10/20/64/128/257 资产，以及全零、常数、重复列、5% 随机 NaN、整列 NaN、1e200 溢出值、`inf`、t 分布厚尾、前后段方差剧变等退化情形。

---

## 五、关于「skfolio 全家崩溃」

### 结论：那是 skfolio 库自身的缺陷，不是我们文件的 bug，我们的文件不受影响

`outputs/round1-audit/crash_diagnosis.py` 可复现全部证据：

| skfolio 估计器 | n=1 | n=2 | n=3 | n=5 |
|---|---|---|---|---|
| `HierarchicalRiskParity` | None | None | ok | ok |
| `HierarchicalEqualRiskContribution` | None | None | ok | ok |
| `SchurComplementary` | None | None | ok | ok |
| `NestedClustersOptimization` | None | None | ok | ok |

两个不同的缺陷，都在 skfolio 内部：

1. **`n_assets == 1`**：`skfolio.distance.PearsonDistance` 返回的是**标量**（shape `()`）而不是 1×1 矩阵，于是 `HierarchicalClustering` 内部的 `pd.DataFrame(...)` 抛出那句极具误导性的一行话 `DataFrame constructor not properly called!`。
2. **`n_assets == 2`**：`squareform` 后的凝聚距离只有一个元素对，`optimal_leaf_ordering` / `linkage` 抛 `attempt to get argmax of an empty sequence`。

**skfolio 的整个层次聚类族只在 `n_assets >= 3` 时有定义。** 而且在 `raise_on_failure=False` 下它会静默吞掉异常、把 `weights_` 设成 `None`、只发一条 `UserWarning`——评分系统拿到 `None` 就记一次失败。占 70% 权重的失败率，就是这么丢的。

### 我们的文件为什么免疫

我们的文件**完全不调用**这些类。它带自己的 HRP 实现（`hrp_weights`），`n == 1` 分支在任何 linkage 之前就返回 1.0。已验证在 skfolio 崩溃的同一批输入上全部产出合法权重：

| 输入 | skfolio 层次族 | 我们的输出 |
|---|---|---|
| n=1 随机游走 | None（失败） | `[1.0]` ✓ |
| n=1 全零 | None（失败） | `[1.0]` ✓ |
| n=1 全 NaN | None（失败） | `[1.0]` ✓ |
| n=1 仅 1 行 | None（失败） | `[1.0]` ✓ |
| n=2 完全共线 | None（失败） | `[1.0, 0.0]` ✓ |
| n=2 等方差 | None（失败） | `[0.503, 0.497]` ✓ |
| n=5 秩亏（5 列同源） | ok | ✓ |

回归测试固化在 `tests/test_edge_robustness.py`（26 项），其中 `test_documented_skfolio_defect_is_still_real` 会在 skfolio 未来修复该缺陷时主动失败提醒，避免这条防护被无声遗忘。

---

## 六、目视复核建议

提交前想自己再确认一次，依次执行：

```bash
cd /Users/shw/Downloads/portfolio-game-round1
export PYTHONPATH=
PY=/Users/shw/.workbuddy/binaries/python/envs/default/bin/python

$PY teacher_reference/self_test.py submission/portfolio_round1.py   # 必须打印 Basic checks passed
$PY tools/verify_repo.py --require-integration                     # 必须退出码 0
$PY -m pytest -q                                                   # 必须 0 failed
```

`--require-integration` 会把老师的 `self_test.py` 和我们的提交文件复制到一个干净的临时目录里执行——不依赖仓库里任何其他代码，这正是评分系统的运行方式。

---

## 七、仍然存在的已知限制（不作隐瞒）

1. **夏普与年化收益在升级后并非全部改善。** sp500 上分别由 0.9820→0.9494、0.1556→0.1442。这是用一部分收益换取更稳健的回撤，判断依据是失败率占 70% 权重的评分结构，不是「新的一定全面更好」。
2. **宽横截面上二次规划常常打不到 1e-8 的相对间隙，此时返回的是树锚而不是锚+倾斜。** 在 1455 资产、252 行这种条件数极差（cond ≈ 2.3e3）的输入上，投影梯度在 600 次迭代内到不了 1e-8（实测相对间隙停在 4.4e-4），代码按设计抛错并退到 HRP 锚。这个退路仍然是用协方差的结构化配置，不是等权也不是逐资产占位，所以降级的是「倾斜」，不是风险模型。放开这一条需要换求解器，代价与收益不成比例，本轮不做。
3. **`n_assets > 2048` 时退为反波动率加权**（`fallback_resource_guard`）。这是成本护栏，实测依据与阈值推导见第四节；2048 以上的数据集不在 skfolio 标准数据集范围内，属于外推场景。
4. **`n_assets == 1` 时权重必然是 `[1.0]`**，这在本轮的约束（多头、满仓、禁止等权）下是数学上唯一的解。文件把这一情形显式标记为 `single_asset_rule_ambiguity: True`，不假装它是一个有选择余地的结果。
5. **回测结论只覆盖 skfolio 自带的 5 个数据集。** 老师明确说过评分数据集不限于示例，所以文档里所有性能数字都标注了数据集与时间段，不宣称对未知测试集的外推能力。

## 八、交付后的追加验证与因子可行性结论（2026-09-19）

本轮追加了三件事，全部围绕"老师的测试集是随机抽股票 × 随机取两年"这个前提：

1. **按出题方式压测。** 新增 `tools/random_window_stress.py` 并接入验收（`verify_repo.py`
   的 `random_windows` 检查项）：5 个数据集、随机子集 × 随机两年窗口 + 早期历史穷举 +
   退化子集，共 **12,132 个折**，失败 0、非法权重 0、等权 0。失败率是 70% 的分数，
   这一项是本次追加验证里权重最高的一条证据。
2. **加固失败回退链。** 候选链末级由"押注单一资产"改为**风险序递减预算**
   （按观测风险升序、按 1/rank 分配后归一化）。它同时满足两个要求：
   **永不等于等权**（本轮明令禁止），且**不会在多标的可用时把全部资金押在一个名字上**。
   新增 `tests/test_equal_weight_prohibition.py`，对 n=1..12、多组窗口长度和一批
   刻意构造的退化面板断言"永不等于 1/n"，并把"最后一级必须分散"钉成回归。
3. **因子倾斜评估后决定不上线。** 从 COS 拉取 A 股 263 个已产出因子，重建了一个
   时点成分 SP500 面板（3,122,695 行 / 5,760 天 / 861 只），按老师口径做纯多回测，
   再把复合信号作为线性倾斜接入现有凸规划。结论是**不上线**：
   在"随机子集 × 随机两年窗口"口径下，所有测试强度下夏普胜率均低于 50%。
   完整取证路径、数据质量发现（COS 复权字段在拆股日未中性化）与数值见
   [round1-factor-transfer-zh.md](round1-factor-transfer-zh.md)。

**对第 1 条的诚实边界**：以上压测覆盖的是 skfolio 自带数据集和自建的 SP500 面板，
不是老师的真实测试集；它证明的是"在能构造出的最接近的场景里不失败、不产出等权"，
不是"隐藏测试一定通过"。
