# Round 1 作业要求 —— 逐句中文翻译 + 硬约束拆解

来源：Canvas 页面 `Portfolio Game Round #1`、`Portfolio Game – Submission Rules & Self-Test`，
以及老师随附的 `self_test.py`、`example_portfolio.py`。
英文原文逐句翻译，不做意译省略。

---

## 一、作业通知原文翻译

> **Portfolio Game Round #1**
> Due Saturday by 1pm ｜ Points 0 ｜ Submitting: a file upload ｜ File Types: py
> 截止：周六下午 1 点（硬性，`Sep 19 at 1pm sharp`）｜ 提交：上传 1 个文件 ｜ 文件类型：`.py`

> Dear MAFS5310 students, Welcome to the Portfolio Game!
> 各位 MAFS5310 的同学，欢迎参加组合投资游戏！

> This email is to request your first portfolio function. Please upload your portfolio function
> in a Python file via Canvas (the deadline is Sep 19 at 1pm sharp).
> 这封邮件是来要你的第一个「组合函数」。请通过 Canvas 上传一个 Python 文件
> （截止时间是 9 月 19 日下午 1 点整）。

> **Note that you are NOT allowed to use the equally weighted portfolio in the portfolio game.**
> **注意：本游戏中不允许使用等权组合。**

> We are going to use the package `skfolio` to evaluate your submitted functions.
> Please make sure your submitted file has followed our instruction.
> 我们将用 `skfolio` 这个包来评估你提交的函数，请务必让你的提交文件符合我们的说明要求。

> **The backtest setting is**
> **回测设定如下**
>
> - `shortselling = FALSE` —— 不允许做空
> - `leverage = 1` —— 杠杆为 1（满仓，不加杠杆）
> - `lookback = 252` —— 用过去 252 个交易日（约 1 年）的数据拟合
> - `optimize_every = 20` —— 每 20 个交易日重新优化一次权重

> **The ranking is based on the weighted average of the rank percentiles of the following
> score quantities:**
> **排名依据是以下几项得分量的「排名百分位」的加权平均：**
>
> | 得分项 | 权重 |
> |---|---|
> | Sharpe ratio（夏普比率） | 10% |
> | max drawdown（最大回撤） | 10% |
> | annual return（年化收益） | 10% |
> | **failure rate（失败率）** | **70%** |

> Enjoy the game!
> 玩得开心！

### 关键解读

1. **不是绝对分数排名，是「排名百分位」加权。** 也就是说：你的夏普是全组第 1 还是第 10
   不重要，重要的是你在全组里的**相对位置**。分数被压缩到 (0,1]，所以某项拿满分也只有 0.1。
2. **失败率独占 70%。** 这是整个游戏最重要的信号：**能跑通、不报错，比赚得多重要 7 倍。**
3. **优化频率 20 天，回看 252 天。** 每次只给过去 252 天收益率，要求输出权重，持有 20 天。

---

## 二、提交规则原文翻译（Submission Rules & Self-Test）

> Please read these rules carefully before submitting your Python portfolio. Two files are
> attached to this page: `self_test.py` (the self-check script) and `example_portfolio.py`
> (a complete, working example).
> 提交前请仔细阅读规则。本页附两个文件：`self_test.py`（自检脚本）和
> `example_portfolio.py`（一个完整可运行的示例）。

> ### Submission rules
> 你的提交是**一个单独的 Python 文件**。它必须满足：
>
> 1. **Define a class named exactly `CVXPYPortfolio`** (case-sensitive, must match exactly).
>    定义一个**名字完全等于 `CVXPYPortfolio`** 的类（大小写敏感，必须一模一样）。
> 2. **Make that class inherit from `skfolio.optimization.BaseOptimization`:**
>    这个类必须继承自 `skfolio.optimization.BaseOptimization`：
>    ```python
>    from skfolio.optimization import BaseOptimization
>    class CVXPYPortfolio(BaseOptimization):
>        ...
>    ```
> 3. **Implement a `fit(self, X, y=None)` method that:**
>    实现一个 `fit(self, X, y=None)` 方法，该方法要：
>    - first calls `validate_data(self, X)` (this converts the returns DataFrame into a numpy array)
>      —— 首先调用 `validate_data(self, X)`（把收益率 DataFrame 转成 numpy 数组）
>    - computes your portfolio weights —— 计算你的组合权重
>    - stores them in `self.weights_` as a **numpy array of shape `(n_assets,)`**
>      —— 存进 `self.weights_`，必须是**形状 `(n_assets,)` 的 numpy 数组**
>    - returns `self` —— 返回 `self`
> 4. **Do not override `__init__`** in a way that rejects the arguments the grading system passes.
>    **不要重写 `__init__`** 到会拒绝评分系统传入参数的程度。
>    （评分系统会传 `portfolio_params={'name': ...}`，还会 clone 这个对象。）
> 5. **Do not contain your own codes on backtesting or downloading data in the submission.**
>    提交文件里**不要包含你自己写的回测代码，也不要包含下载数据的代码**。
> 6. **The file name itself can be anything** (e.g. your student ID), but the class inside must
>    still be named `CVXPYPortfolio`.
>    文件名随便起（比如你的学号），但里面的类名必须还是 `CVXPYPortfolio`。
>
> `example_portfolio.py` shows all of this in a minimum-variance example. Use it as a template.
> `example_portfolio.py` 用最小方差这个例子演示了以上全部要求，可以拿它当模板。
>
> **Depending on the round, there may be additional portfolio constraints (e.g. fully invested,
> short selling, leverage). Follow the round brief for those.**
> 不同轮次可能会有额外的组合约束（比如必须满仓、禁止做空、杠杆限制），以当轮说明为准。

### 自检（Self-test before submitting）

> `self_test.py` checks the basic format above and runs a quick backtest on skfolio's built-in dataset.
> `self_test.py` 会检查上面这些基本格式，并在 skfolio 内置数据集上跑一个快速回测。

> Run it from the terminal: `python self_test.py your_portfolio.py`
> 在终端运行：`python self_test.py your_portfolio.py`

> - If it prints `Basic checks passed` followed by a summary, your file meets the basic requirements.
>   如果打印出 `Basic checks passed` 并跟着一段摘要，说明你的文件满足基本要求。
> - If it raises an error, read the message and fix the problem before submitting.
>   如果报错，读错误信息，提交前先修好。

> **Important:** the self-test only checks the basic structure — class name, inheritance,
> constructor, `fit`, and the format of `weights_` (correct shape, finite values).
> It does **not** check round-specific rules such as short selling, leverage, or being fully
> invested; those are enforced by **the grading system**.
> **重要：** 自检只查基本结构——类名、继承、构造函数、`fit`、`weights_` 的格式
> （形状对不对、值是否有限）。它**不检查**本轮的具体规则（做空、杠杆、是否满仓），
> 这些由**评分系统**强制执行。

> Please run the self-test on your own file and make sure it passes before you submit.
> 请务必对你的文件跑一次自检，确认通过再提交。

### 「failure rate」到底是什么意思？

> The failure rate is not a performance measure. It has nothing to do with whether your
> portfolio makes or loses money, beats a benchmark, or reaches a particular return target.
> 失败率**不是业绩指标**。它跟你赚不赚钱、跑不跑得赢基准、有没有达到某个收益目标**毫无关系**。

> It measures the **robustness and validity** of your submitted function when it runs on
> **many different test datasets**. A failure is recorded if the function **cannot produce a
> valid portfolio** on a test dataset — for example because of a runtime error, a numerical
> issue, an optimization failure, or invalid weights that do not satisfy the required specifications.
> 它衡量的是：当你的函数在**许多不同的测试数据集**上运行时，它的**稳健性和有效性**。
> 只要函数在某个数据集上**无法产出一个合法的组合**，就记一次失败——比如运行时错误、
> 数值问题、优化失败，或者算出来的权重不满足规定要求（非法权重）。

> The failure rate is: `Number of failed test datasets / Total number of test datasets`
> 失败率 = 失败的测试数据集数量 / 测试数据集总数

> **The test datasets used for grading are not limited to the examples given during development.
> Your function should be designed to handle different datasets robustly and consistently.**
> **评分用的测试数据集不限于开发期间给出的示例。你的函数必须被设计成能稳健、一致地
> 处理不同的数据集。**

### 关键解读

- 「失败」的判定范围比想象的宽：**任何异常 → 失败，优化不收敛 → 失败，
  权重不合法（负数、不满足满仓、shape 不对、NaN/Inf）→ 也失败。**
- 老师明确说了**测试集不止示例里那一个**。所以要考虑：资产数从 1 到上千、
  样本长度从几十行到上万行、不同国家市场、不同起始时间。
- `self_test.py` 用的是 `load_sp500_dataset()`（20 只股票，1990–2022）；
  但评分用的数据集可能包括 skfolio 的 `ftse100`（64 只）、`nasdaq`（1455 只）、
  `sp500_index`（1 个资产）、`factors`（5 个因子）等。

---

## 三、硬约束清单（可直接当验收表用）

| # | 约束 | 违反的后果 |
|---|---|---|
| 1 | 文件是一个 `.py`，只能有 `numpy`/`sklearn`/`skfolio`/`scipy` 等已装依赖 | 导入失败 → 直接 0 分 |
| 2 | 类名精确为 `CVXPYPortfolio`（区分大小写） | 找不到类 → 失败 |
| 3 | 必须 `issubclass(CVXPYPortfolio, skfolio.optimization.BaseOptimization)` | 类型错误 → 失败 |
| 4 | `fit(self, X, y=None)`，第一步 `validate_data(self, X)` | 格式不符 → 失败 |
| 5 | `self.weights_` 是 `(n_assets,)` 的 numpy 数组，全部有限（无 NaN/Inf） | 失败 |
| 6 | 权重全部 ≥ 0（`shortselling=FALSE`；**注意 1e-9 级的负噪声也会被判非法**） | 失败 |
| 7 | `sum(weights) == 1`（`leverage=1`，满仓） | 失败 |
| 8 | **不能等于等权**（`1/n_assets`） | 明确禁止 |
| 9 | 不重写 `__init__`（评分系统会传 `portfolio_params`，且会 `clone`） | 失败 |
| 10 | 文件内不写回测、不下载数据、不读写文件、不联网 | 违规 |
| 11 | 只允许用 `X`（历史收益率）里的信息，不得依赖上次 `fit` 的残留状态 | 违规/失败 |
| 12 | 必须能处理：资产数 = 1、资产数很大、含 NaN、窗口很短等边界 | 每次失败扣 70% 权重 |

---

## 四、数据入口（回答"数据从哪来"）

**结论：老师给的材料里确实写明了数据入口，就是老师的示例脚本里那一行。**

`self_test.py` 第 5 行附近：

```python
from skfolio.datasets import load_sp500_dataset
X = load_sp500_dataset().pct_change().dropna()
```

- 数据来自 **`skfolio` 包自带/可下载的数据集**，不是单独的 Excel/CSV。
- 传给 `fit` 的 `X` 是**收益率**（`pct_change` 之后的 DataFrame），
  **不是价格**。这正是你说的「好像只给了收益率」。
- 内置无需下载的：`load_sp500_dataset()`（20 资产，1990-01-02~2022-12-28）、
  `load_sp500_index()`（1 个资产）、`load_factors_dataset()`（5 个因子，2014 起）
- 需要联网下载并缓存的：`load_ftse100_dataset()`（64 资产，2000~2023-05）、
  `load_nasdaq_dataset()`（1455 资产，2018-01~2023-05）、
  `load_sp500_implied_vol_dataset()`

**关于「能不能把收益率还原成 close 再算因子」**：可以，但信息量有限。
`close_t = cumprod(1 + r_t)` 只能还原出一条**归一化的价格路径**（起点任意），
所以：
- ✅ 能算：动量/趋势（累计收益、MA 斜率）、波动率、回撤、自相关——
  这些都只依赖路径形状，对起点不变。
- ❌ 不能算：任何依赖**绝对价格水平**的指标（如真实价格、市值、面值）；
  也不能做**跨资产价格比较**，因为每条路径都被强行归一到同一个起点。
- ⚠️ 而且本轮的评分口径里，`X` 在每个窗口都是**过去 252 天收益率**，
  再"还原价格"≈ 在窗口内重新累计，等价于对收益率做加权，收益有限。
  结论：这一轮先不做因子是对的，**把 70% 的失败率吃干榨净更划算**。
