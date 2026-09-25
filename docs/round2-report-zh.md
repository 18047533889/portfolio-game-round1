# Round 2 交付报告（MAFS5310 Portfolio Game 第二轮）

- 截止：2026-09-26 13:00
- 提交物：`submission/portfolio_round2.py`（**单文件**）
- SHA256：`9c38aa47cd1ba82bb4245f216def87a89c7baca86c20faee317f4bf55c6c3319`
- 610 行 / 27,672 字节

---

## 一、规则变化与它带来的两个后果

| 项目 | Round 1 | Round 2 |
|---|---|---|
| 禁止的组合 | 等权（EWP） | **EWP + 逆波动率（IVP）+ 用样本协方差矩阵的全局最小方差（SCM-GMVP）** |
| shortselling | FALSE | **TRUE** |
| leverage | 1 | 1 |
| lookback / optimize_every | 252 / 20 | 252 / 20 |
| weight_drift | 未提及（默认 False） | **TRUE**（要求 skfolio ≥ 1.3.0） |
| 评分 | Sharpe 10% + 回撤 10% + 年化 10% + **失败率 70%** | 年化 **15%** + 回撤 **15%** + **失败率 70%** |

后果一：**Sharpe 被移出评分，年化权重翻倍并追平回撤。** Round 1 的 `anchor_penalty=0.25`
是在「回撤在 8/9 个评价块中改善」这个判据下选的，而那个判据诞生于 Sharpe 还占 10% 的语境。
本轮必须**重新推导**这个参数，不能继承。

后果二：**`weight_drift=TRUE` 会击穿 skfolio 的层次聚类族。** 见第三节，这是本轮最重要的发现。

---

## 二、「允许做空」与「杠杆为 1」的关系，以及我们的选择

邮件同时写了 `shortselling = TRUE` 和 `leverage = 1`。skfolio 库里没有叫 `leverage` 的参数
（`leverage` 在库内只是财务杠杆**因子**），所以这两个是评测框架自己的规则。它们怎么组合，
决定做空有没有空间。

skfolio 对敞口的定义是明确的：`Long`=正权重和，`Short`=负权重和，**`Net`=权重和，
`Gross`=绝对权重和**。满仓（`sum(w)=1`）时恒有

```
Net   = 1
Gross = 1 + 2 × Short
```

也就是**"可做空"与"`Gross ≤ 1`"在数学上互斥**。据此穷举 `leverage=1` 的四种合理解读：

| 解读 | 约束 | 做空空间 |
|---|---|---|
| ① Gross 上限 + 强制满仓 | `Gross ≤ 1` 且满仓 `Net = 1` | 无（数学上只有 w≥0 成立） |
| ② 真杠杆率 | `Gross / Net = 1` | 无（同上） |
| ③ Gross 上限（不强制满仓） | `Gross ≤ 1` | 有，但净敞口 `= 1 − 2×Short` 被压低，收益吃亏 |
| ④ 只约束净敞口 | `\|Net\| ≤ 1` | 有，且可保持满仓 |

**long-only 满仓在 ① ② ③ ④ 下全部合法**；带做空的书只在 ③ 或 ④ 下合法。
**做空因此是一个不对称赌注**：开启它只会增加"被判违规"的面数，而失败率占 70% 的分。

我们提交的是 long-only 满仓（`w ≥ 0, sum(w) = 1`）。做空**已经作为参数实现并通过扫描**
（`allow_short` / `short_cap`，出厂关闭），完整证据与手调比例的结果见
**[docs/round2-short-selling-zh.md](round2-short-selling-zh.md)**。一句话概括：
在出厂 `anchor_penalty=4.0` 下开做空，年化与回撤的变动都可忽略（−0.04pp / −0.47pp），
而 Gross 从 1.000 抬到 1.156；模拟评分 0.509–0.516，而**任一折失败 = −0.51**。

---

## 三、本轮最重要的发现：`weight_drift=TRUE` 会让 skfolio 的 HRP / HERC 全数失败

本轮把 `weight_drift` 打开后，skfolio 自带的层次聚类优化器出现 **100% 级别的失败**：

| 数据集 | `HierarchicalRiskParity` | `HierarchicalEqualRiskContribution` |
|---|---|---|
| sp500(20) | **402 / 403 折失败** | **402 / 403 折失败** |
| ftse100(64) | **282 / 283 折失败** | **282 / 283 折失败** |
| factors(5) | **99 / 100 折失败** | **99 / 100 折失败** |

根因（逐层定位到源码行）：

```
ValueError: If `previous_weights` is provided as a dictionary,
you must input `X` as a DataFrame with assets names in columns
```

`weight_drift=True` 时 `cross_val_predict` 会把上一折的权重**以字典形式**传给下一折，
而 skfolio 的 HRP/HERC 在 `_base.py::_risk()` 里拿这个字典去构造 `Portfolio`，
此时 `X` 已被转成 numpy，列名丢失 → 每折必抛。

这对本轮的策略含义很大：**任何直接提交 `HierarchicalRiskParity` 或
`HierarchicalEqualRiskContribution` 的同学，failure rate 会接近 100%，而失败率占 70% 的分。**
我们自己的实现不碰 `previous_weights`（权重不用它算成本，也不传给 Portfolio），所以完全不受影响。

---

## 四、方法：沿用 Round 1 的框架，重推参数并换掉不合规的兜底

**目标函数**（在 `w ≥ 0, sum(w) = 1` 上最小化）：

```
w'Cw / (b'Cb)  +  penalty · ||w − b||² / ||b||²
```

其中 `b` 是 HRP 锚，`C` 是收缩并去噪后的协方差。求解器是带重启的加速投影梯度法，
用**凸对偶间隙**作为收敛证书，不是「可行即收敛」。

**估计链**：Ledoit-Wolf 收缩 → Marchenko-Pastur 特征值去噪 → EWMA 短期波动 → HRP 锚 → QP。

### 参数重推：`anchor_penalty` 0.25 → 4.0

在 Round 2 口径下（`weight_drift=True`，`WalkForward(252,20)`）扫描，
并用**排名百分位模拟**（对手池 = skfolio 的各优化器）评估：

| 配置 | sp500 年化 | sp500 回撤 | ftse100 年化 | ftse100 回撤 | factors 年化 | factors 回撤 |
|---|---|---|---|---|---|---|
| p=0.25（Round 1 交付） | 14.10% | 48.47% | 13.34% | 41.40% | 10.84% | 39.38% |
| p=1 | 15.16% | 54.02% | 13.38% | 43.52% | 10.94% | 39.72% |
| **p=4（本轮交付）** | 16.41% | 57.00% | 13.68% | 45.67% | 10.93% | 39.87% |
| p=16 | 17.13% | 58.51% | 13.96% | 46.40% | 10.91% | 39.92% |
| 等权参照（禁） | 18.24% | 57.03% | 11.78% | 58.79% | 11.16% | 41.26% |

两个结论决定了取值：

1. **70% 的失败率压倒一切。** 所有 0 失败候选的得分挤在 0.51–0.60 的窄带里，
   而 HRP/HERC 因 99% 失败只得 **0.0233**。年化与回撤合计 30% 在有效前沿上几乎没有区分度
   —— 这正是 Round 1 配对检验结论在 Round 2 口径下的重现。
2. **在「排名近似打平」时按合规距离取参数。** 三天集平均，p=4 略高；更重要的是
   p 越大离三个被禁组合越远。p=0 被**明确排除**（见下节）。

### 兜底链重做（这是本轮必须改的合规项）

Round 1 的回退链里有六处调用 `_inverse_risk()` —— 那是 **w ∝ 1/σ，就是本轮被禁的 IVP**。
本轮的处理是**删除，不是重新加权**：

| 层级 | Round 1 | Round 2 |
|---|---|---|
| 主路径 | QP 解 | QP 解（不变） |
| 次级 | HRP 锚 | HRP 锚（不变） |
| 末级 | 逆波动率预算（= IVP） | **按列序线性递减预算**（与任何风险估计无关） |

新的末级规则刻意**不使用风险信息**：在风险无法识别的退化输入上（零方差、重复列、常数列），
一个「按风险排序的预算」不过是把 IVP 读成序数，不值得辩护。线性递减是一个**约定的确定性规则**，
而不是伪装成估计的东西。代价是：在这种输入上输出依赖列序 —— 因为输入本身不含任何能打破对称的信息。
该分支对「至少两列方差为正」的任何输入都不可达，而测试中观察到的每一个面板都满足这一点。

---

## 五、三条禁令的合规论证

| 禁令 | 为什么我们不违反 | 可核验证据 |
|---|---|---|
| **EWP** | 没有任何代码路径在 n>1 时返回 1/n；`check_weights` 会拒绝，`_finish` 会跳过 | 12,000 折压测 `equal=0`；23 例合成退化 `equal-weight outputs = 0` |
| **IVP** | 没有任何代码路径形成 1/σ；Round 1 的 `_inverse_risk` 已**删除** | 与 IVP 的相关性：p=4 时 sp500 r=+0.406、ftse100 r=+0.256、factors r=+0.483 |
| **SCM-GMVP** | 协方差经 Ledoit-Wolf 收缩 + MP 去噪，目标函数带锚正则项，不等于 `argmin w'Sw` | 与 SCM-GMVP 的相关性：p=4 时 r=+0.628（sp500），远离 p=0 的 r=+0.977 |

**关于 p=0 的诚实记录**：`anchor_penalty=0` 会让方法**退化成本轮被禁的 SCM-GMVP** —
在 5 资产面板上 MP 去噪被跳过（`T ≤ N+2`）、Ledoit-Wolf 在 252 样本下收缩很弱，
结果与 SCM-GMVP 的相关性 **0.999、L1 距离 0.054**。那不是一个「相似」的方法，那就是同一个方法。
这条发现同时推翻了我原先「因为我们用的是收缩协方差所以天然合规」的说法：**在低维数据上，
这个辩解不成立，真正让我们避开禁令的是 p>0 的锚正则项。** 因此 p=0 不交付。

---

## 六、验证结果

| 检查 | 结果 |
|---|---|
| 老师原版 `self_test.py` | `Basic checks passed`、**127 组合、0 失败、0 回退** |
| **隔离测试**（空目录，仅本文件 + `self_test.py`） | 通过：127 组合、0 失败、0 回退（证明无仓库内隐式依赖） |
| 合成退化压力（`tools/round2_stress.py`，23 例敌意面板） | **0 失败 / 0 非法 / 0 等权 / 0 逆波动率**；`regularized_qp` 17 例、`fallback_hrp` **0** 例 |
| **随机子集 × 随机两年窗口**（`tools/round2_random_window.py`） | 4 数据集、480 切片、**12,000 折 / 0 失败 / 0 非法 / 0 等权**；每折 `Gross ≤ 1` |
| 出厂文件回测（`tools/round2_backtest_report.py`） | 786 组合、**0 失败 / 0 回退**、**精确命中被禁组合 0 次**、`Gross_max = 1.000` |
| 做空投影与对偶间隙下界（`tools/round2_short_projection_check.py`） | 与独立参照（二分 / cvxpy）一致到 **1.4e-15** |
| **加参数未动老路径**（`tools/round2_long_only_equivalence.py`） | 220 个 case、**0 差异**（`np.array_equal`）+ 10 个可证伪的「有意修复」 |
| `pytest` 全套 | **128 通过 / 1 跳过 / 0 失败** |
| 与 `core.py` 及冻结配置一致性 | `build_submission_round2.py --check` 通过 |

随机窗口压测覆盖：子集尺寸偏向 1–25 只（小集合是风险模型最容易崩的地方）、
窗口起点有 40% 概率落在历史前 30%（模拟「很可能是非常非常早的年份」）、
每个折单独校验形状/有限/满足做空下限/满仓/非等权。

---

## 七、必须说清的边界

1. **「0 失败」不等于「隐藏测试一定通过」。** 上面覆盖的是 skfolio 自带数据集，
   不是老师的真实测试集。它证明的是：在能构造出的最接近场景里不失败、不产出任何被禁组合。
2. **回合间是「用收益换回撤」的交换，不是效率提升。** 沿有效前沿移动 p 时，
   年化与回撤同向变化、量级相当；配对检验（Round 1）显示排名改善的块占比接近抛硬币。
   选 p=4 是在这个前提下按合规距离做的取舍，不代表它在该前沿上更优。
3. **`leverage` 的语义未能从邮件本身完全确定。** 我们的 long-only 满仓在四种解读下都合法，
   所以这个不确定性被规则设计本身吸收了，而不是被猜中。
4. **「允许做空」这个自由度：选项已实现、已验证、已手调扫描，但出厂关闭。**
   这不是忽略邮件，而是测量后的取舍：在出厂惩罚系数下开做空对年化/回撤的影响都可忽略、
   却把 Gross 抬到 1.156，而模拟评分里这一项最多值 0.007、任一折失败值 −0.51。
   完整证据见 [docs/round2-short-selling-zh.md](round2-short-selling-zh.md)。
5. **`short_cap` 是每资产上限，不是组合做空比例**，两者随资产数不同而分离
   （`cap=0.01`：sp500 实际做空 0.057，ftse100 0.176）。该边界是有意保留的，见做空专文 §6。
6. **做空工作顺带修掉的两个既有缺陷改变了旧核心在边缘面板上的行为**，
   所以「逐位一致」严格成立于 210/220 个 case，另外 10 个以可证伪的方式单独断言。

---

## 八、复现命令

```bash
python tools/build_submission_round2.py          # 从 core 生成提交文件
python tools/build_submission_round2.py --check  # 校验已同步
python teacher_reference/self_test.py submission/portfolio_round2.py
python tools/round2_stress.py                    # 合成退化压力
python tools/round2_random_window.py --reps 120  # 老师出题口径压测
python research/round2/rank_sim.py               # 排名百分位模拟
```

研究脚本与原始结果见 `research/round2/` 与 `reports/round2_*.{json,txt}`。
