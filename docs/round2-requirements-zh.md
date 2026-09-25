# Round 2 作业要求 —— 逐句中文翻译 + 硬约束拆解

来源：老师 Round 2 通知邮件原文（2026-09 发出），以及邮件中提到的
"regularly updated instructions page"。
英文原文逐句翻译，不做意译省略。Round 1 的对应文档见
`docs/round1-requirements-zh.md`。

---

## 一、通知原文翻译

> Dear MAFS5310 students,
> 各位 MAFS5310 的同学，

> This email is to request your second portfolio function.
> 这封邮件是来要你们的**第二个**组合函数。

> Please upload your portfolio function in a Python file via Canvas
> (the deadline is **Sep 26 at 1pm sharp**).
> 请通过 Canvas 上传你的组合函数（Python 文件），**截止时间是 9 月 26 日下午 1 点整**。

> Note that you are **NOT allowed** to use the following portfolios in this round:
> 注意，**本轮不允许**使用以下组合：

> - **equally weighted portfolio (EWP)** —— 等权组合
> - **inverse-volatility portfolio (IVP)** —— 逆波动率组合
> - **global minimum variance portfolio (GMVP) using sample covariance matrix (SCM)**
>   —— 使用**样本协方差矩阵**的全局最小方差组合

> The backtest setting is
> 回测设定如下：

> - `shortselling = TRUE` —— **允许**做空（注意：是「允许」，不是「必须」）
> - `leverage = 1` —— 杠杆为 1
> - `lookback = 252` —— 用过去 252 个交易日拟合
> - `optimize_every = 20` —— 每 20 个交易日重新优化
> - `weight_drift = TRUE`（please upgrade skfolio to the latest version `>= 1.3.0`
>   to enable this setting）—— 持仓期内权重随价格**漂移**；
>   **请把 skfolio 升级到 ≥ 1.3.0 才能启用这个设定**

> The ranking is based on the weighted average of the rank percentiles of the
> following score quantities:
> 排名依据是以下几项得分量的**排名百分位**的加权平均：

> - **annual return** —— 年化收益：**15%**
> - **max drawdown** —— 最大回撤：**15%**
> - **failure rate** —— 失败率：**70%**

> Enjoy the game!
> 祝玩得开心！

> PS: For helpful guidance, please feel free to consult our regularly updated
> instructions page, which may address your question.
> 附：如果需要指引，可以查阅我们**定期更新**的说明页面，你的问题可能已经在上面解答了。

---

## 二、与 Round 1 的差异（这才是本轮真正的题目）

| 项目 | Round 1 | Round 2 | 后果 |
|---|---|---|---|
| 禁止组合 | EWP 一个 | **EWP + IVP + SCM-GMVP 三个** | Round 1 的兜底链直接踩线，必须重做 |
| `shortselling` | FALSE | **TRUE** | 新增自由度（见第四节） |
| `leverage` | 1 | 1 | 未变 |
| `lookback` / `optimize_every` | 252 / 20 | 252 / 20 | 未变 |
| `weight_drift` | 未提及（默认 False） | **TRUE** | **改变了 skfolio 的代码路径**，不是只改收益算法 |
| 评分项 | Sharpe 10% + 回撤 10% + 年化 10% + 失败率 70% | 年化 **15%** + 回撤 **15%** + **失败率 70%** | Sharpe 出局，年化翻倍并追平回撤 |

两条最需要指出的：

1. **Sharpe 被移出评分。** Round 1 的参数是在「Sharpe 占 10%」的语境下选的，
   本轮必须**重新推导**，不能继承。
2. **`weight_drift = TRUE` 不只是换收益算法**，它让 skfolio 在折与折之间传递
   `previous_weights`，从而改变被调用的代码路径。见 `docs/round2-report-zh.md` 第三节。

---

## 三、三条禁令的准确边界

**「GMVP using SCM」这个措辞里，限定词是 SCM（样本协方差矩阵），不是 GMVP 本身。**
这是一个必须抠清楚的边界：禁令针对的是「用样本协方差矩阵求出来的最小方差组合」，
而不是「任何最小方差思路」。但**这不能当成擦边球的许可**——
真正决定合规的是输出与禁令组合的**可测距离**，而不是目标函数怎么写。
Round 2 的实测发现：在 5 资产面板上，`anchor_penalty=0` 的输出与 SCM-GMVP 的
相关性高达 **0.999**、L1 距离只有 **0.054**——那已经不是「相似」，是同一个东西。
详见 `docs/round2-report-zh.md` 第五节。

同样地，**IVP 是一条结构性禁令**：只要代码里存在「把列方差变成 1/σ 权重」的规则，
无论它出现在主路径还是兜底路径，都是违规。Round 1 的回退链里有六处这样的调用。

---

## 四、`shortselling = TRUE` 与 `leverage = 1` 的组合含义

邮件同时给了这两个设定，但 skfolio 库中**并不存在**名为 `leverage` 的参数
（库内 `leverage` 只是财务杠杆类**因子**），所以这两个是评测框架自己的规则。

skfolio 对敞口的定义是明确的（`MultiPeriodPortfolio.long_short_exposure`）：

- `Long` = 正权重之和
- `Short` = 负权重之和
- `Net` = 所有权重之和
- `Gross` = 权重绝对值之和

据此穷举 `leverage = 1` 的三种合理解读：

| 解读 | 约束 | 做空空间 |
|---|---|---|
| Gross 上限 + 强制满仓 | `Gross ≤ 1` 且 `Net = 1` | 无（数学上只有 `w ≥ 0` 成立） |
| 真杠杆率 | `Gross / Net = 1` | 无（同上） |
| Gross 上限，不强制满仓 | `Gross ≤ 1` | 极窄，且净敞口被压低，收益吃亏 |

**三种解读都指向 long-only。** 提交 long-only 满仓（`w ≥ 0, sum(w) = 1`）
在**任何一种解读下都合法**：`Net = 1` 满足，`Gross = 1` 也满足。
不确定性被规则设计本身吸收了，而不是被猜中。

---

## 五、提交格式（与 Round 1 相同，未变）

- 文件名任意，**单个 `.py` 文件**
- 必须定义**类名完全相同**的 `CVXPYPortfolio`
- 必须继承 `skfolio.optimization.BaseOptimization`
- 必须实现 `fit()` 并设置 `self.weights_`（形状 `(n_assets,)` 的 numpy 数组）
- **不要**覆写 `__init__`（评测端用 `cls(portfolio_params={...})` 实例化）
- 评分时调用形式：`model.raise_on_failure = False` 后 `fit(X)`

老师的原版自检脚本在 `teacher_reference/self_test.py`，
它**只检查格式**，明确声明不检查 leverage / shortselling 等分轮规则
（"It does NOT check leverage, short selling, or any other extra rules specific
to each portfolio game round"）——那些由评分系统按轮次执行。

---

## 六、本轮真正的胜负手

失败率占 **70%**。在有效前沿上挪动参数时，年化与回撤同向变化、量级相当，
排名百分位在合规候选之间几乎没有区分度——**0 失败的合规候选全部挤在
0.51–0.60 的窄带内，而失败率接近 100% 的方法只有 0.0233。**

所以本轮的最优策略是：**不要弄坏任何东西**。
这与 Round 1 配对检验得到的结论是同一个结论，只是在另一套评分规则下重现。
