> **本次状态更新（2026-09-19）：** 请先阅读 [docs/round1-final-report-zh.md](docs/round1-final-report-zh.md) 与 [reports/RUN_REPORT.md](reports/RUN_REPORT.md)。
>
> 真实 skfolio 验收已通过：`tools/verify_repo.py --require-integration` 返回 0，`pytest` 108 项（107 通过 / 1 跳过 / 0 失败），老师原版 `self_test.py` 打印 `Basic checks passed`、127 个组合、**0 个失败组合**，`ready_for_teacher_submission: true`。240 组合成边界数据压力测试最终失败 0。
>
> **模拟老师的出题方式**（随机抽股票子集 × 随机两年窗口）已单独压测：5 个数据集、**12,132 个折**，失败 0、非法权重 0、等权 0。回归入口在 `tools/verify_repo.py` 的 `random_windows` 检查项。
>
> 失败回退链已加固：候选链末级由"押注单一资产"改为**风险序递减预算**（按观测风险升序、按 1/rank 分配再归一化），既永远不等于等权，也不会在多标的可用时把全部资金押在一个名字上。回归测试见 `tests/test_equal_weight_prohibition.py`。
>
> 算法已升级：相关矩阵增加 **Marchenko–Pastur 特征值去噪**，锚惩罚由 1.0 调整为 **0.25**。选型依据见 [docs/method-selection-zh.md](docs/method-selection-zh.md)。
>
> **因子倾斜经完整评估后决定不上线**：从 COS 拉取 A 股 263 个已产出因子，移植到美股时点成分 SP500 面板回测，再作为线性倾斜接入优化端。在"随机子集 × 随机两年窗口"口径下，所有测试强度下夏普胜率均低于 50%。完整证据见 [docs/round1-factor-transfer-zh.md](docs/round1-factor-transfer-zh.md)。
>
> GitHub main 已包含完整代码。上次上传失败的实际原因是 HTTPS OAuth token 缺少 `workflow` 权限、无法推送 `.github/workflows/tests.yml`（并非平台安全检查）；改用 SSH 推送后成功，原占位提交保留在 `backup/pre-upload` 分支。下文是原项目使用说明。

# Portfolio Game Round 1｜组合优化作业与研究工具

MAFS5310 第一轮。只使用老师提供的资产收益率，不使用额外因子、外部行情、预训练模型或虚构现金资产。

> **本次交付状态：代码、数值测试与真实 skfolio 验收均已完成，`submission/portfolio_round1.py` 可直接提交。**
> 最新实际运行记录见 `reports/verification.json`、`reports/preflight.txt` 与 `docs/round1-final-report-zh.md`。
> 合成数据测试不是市场回测成绩，二者在文档中始终分开陈述。

## 1. 哪一个给老师？

**只上传 `submission/portfolio_round1.py` 这一个文件。**

它定义 `CVXPYPortfolio(BaseOptimization)`，继承父类构造函数，`fit(self, X, y=None)` 首先调用 `validate_data`，将形状为 `(n_assets,)` 的 NumPy 权重写入 `self.weights_`，返回 `self`。

这个文件已经包含全部算法，不需要带上 `src/`、`research/` 或 JSON 配置。它不下载数据、不读写文件、不自行运行回测。完整验收前，不要把它视为已获得提交保证的版本。

`teacher_reference/example_portfolio.py` 是老师的原始样例，**不是你的提交文件**。原样例优化失败后会退到等权，与本轮规则冲突，因此不能直接提交它。

## 2. 目录分工

```text
submission/portfolio_round1.py     给老师：唯一、独立的提交文件
src/portfolio_game/core.py         内部：算法的唯一源代码
configs/submission.json            内部：当前冻结参数，不由提交文件读取
research/backtest.py               内部：一个冻结方案的向前回测
research/tune.py                   内部：有限参数搜索，隔离保留段
research/models.py                 内部：主方案、HRP、反波动率、可选 Schur 对照
research/data.py                   内部：公开样本或本地 CSV 入口
research/engine.py                 内部：持仓漂移、费用、失败窗口记录
research/metrics.py                内部：完整收益序列上的指标
teacher_reference/                老师给的两份 .py，保持原样并保存哈希
tests/                            数值、构建、时间隔离、真实库集成测试
tools/build_submission.py         将同一算法原文嵌入独立提交文件
tools/preflight.py                真实 skfolio + 原版自测；缺依赖直接失败
tools/verify_repo.py               重跑验证并生成 JSON 证据
tools/stress.py                    合成压力测试，不是收益表现证据
tools/publish_github.sh            在本人已登录的 GitHub CLI 上创建私有仓库并上传
reports/                          真实测试记录和未完成事项
```

**不要分别维护两套算法。**内部修改 `core.py`，通过构建工具更新提交文件；一致性测试会检查嵌入代码和参数是否过期。

## 3. 当前算法

252 期历史收益 → Ledoit–Wolf 收缩得到长期相关矩阵 → **Marchenko–Pastur 特征值去噪** → 近期波动/长期相关性混合 → 树结构 HRP 参考配置 → 正则化最小方差。

去噪这一步只动「与抽样噪声无法区分」的那部分谱：低于 MP 上边缘的特征值被合并到它们的共同均值，再重建并归一化回单位对角。它不改变被估计的对象，只降低估计的方差；当 `n_assets < 2` 或 `n_observations <= n_assets + 2` 时直接跳过，而不是做一个不可靠的近似。

目标为：

\[
\min_{w\geq0,\;\mathbf 1^Tw=1}
\frac{w^T\Sigma w}{b^T\Sigma b}
+\lambda\frac{\|w-b\|_2^2}{\|b\|_2^2}.
\]

这里 `b` 是 HRP 参考权重。采用单纯形投影梯度算法求解同一个凸二次规划，并用一阶对偶间隙检查最优性；达到迭代上限而未通过误差检查时，明确记录降级，不把仅仅“有权重”称作优化成功。

类名按照老师要求叫 `CVXPYPortfolio`，但规则没有要求求解器必须是 CVXPY。使用这一求解方式是为了不额外依赖某个特定商业或可选求解器。CVXPY 仅作为可选的独立数值对照；skfolio 本身的依赖仍需正常安装。

当前参数 `half_life=63`、`recent_mix=0.25`、`anchor_penalty=1.0` 是事前设计默认值，**未进行真实市场调参，不能称为最优参数**。

完整解释见 `docs/algorithm.md`。

## 4. 安装与真正的提交前检查

建议使用独立的 Python 3.11–3.13 环境。当前实际测试环境是 Linux/Python 3.13.5；其他版本由 CI 配置覆盖，但本次没有实际运行远端 CI。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python tools/build_submission.py --check
python tools/verify_repo.py --require-integration
```

Windows 使用 `.venv\Scripts\activate`。系统只有 `python3` 时，相应替换命令中的 `python`。

正式自测也可单独运行：

```bash
python tools/preflight.py
```

该命令先运行真实 skfolio 的导入、构造、克隆、fit/predict、252/20 与 252/63 检查，再将**独立提交文件**和老师原版 `self_test.py` 复制进临时目录执行。它不会修改老师原版脚本。

缺少 skfolio 时会显示 `BLOCKED` 并返回非零退出码。直接运行 `pytest` 在缺依赖时会跳过真实库测试模块；因此，**不能只看 pytest 没报错就提交**。

本轮权重约束检查还包括：只做多、满仓、不加杠杆、排除等权。原版自测不负责全部本轮约束。教师所用版本、权重容差和隐藏数据仍然未知。

## 5. 我们自己如何回测与调参？

默认数据入口与老师一致：`load_sp500_dataset()` 的价格转换为简单收益率。公开样本的过去与未来按时间隔离，不用随机打乱验证。

```bash
# 不涉及调参：评估当前固定配置的 2013–2017 验证区间
python -m research.backtest --phase validation --output outputs/validation

# 14 个固定候选；2018 年及之后不进入参数选择
python -m research.tune --output outputs/tuning

# 增加真正的 skfolio Schur 对照（依赖缺失/不合规时记录失败，不冒充别的方法）
python -m research.tune --include-schur --output outputs/tuning-with-schur

# 对开发期筛选出的少量方案再比较半衰期 42/126
python -m research.tune --refine-half-life --output outputs/tuning-refined
```

先固定半衰期 63，比较近期权重 `{0,0.25,0.5}` 和正则化 `{0,0.3,1,3}` 的 12 个组合，另有 HRP 与反波动率两个对照。`rho=0, lambda=0` 本身就是收缩最小方差，所以不重复计入同一方案。

开发期截至 2012-12-31；验证期是 2013–2017；2018 年以后保留。14 个候选先在开发期筛选，少数候选进入验证期。半衰期细化是有边界的可选动作，不在老师的每次 `fit()` 中搜索。

内部选择优先排除失败或频繁降级的方案，再比较逐年夏普、收益和回撤的排名百分位。收益排名默认使用 CAGR（复合年化收益率），同时报告年化算术平均；教师未明确“annual return”的确切计算方式，可用 `--return-metric annualized_mean` 做敏感性检验。内部排名不是老师正式成绩预测。

有自己的数据时，明确区分价格和收益率；CSV 第一列为递增、无重复的日期，其他列是资产：

```bash
python -m research.backtest --returns-csv data/returns.csv --phase validation
python -m research.tune --prices-csv data/prices.csv --output outputs/tuning-local
```

本地数据日期必须覆盖所用开发/验证区间。需要使用其他年代时，应先修改并记录 `research/data.py` 的日期切分，而不是根据保留段成绩挑切分点。

### 冻结与导出

调参仅写入 `outputs/tuning/selected.json`，**不会自动修改老师文件**。确认选择合理后：

```bash
cp outputs/tuning/selected.json configs/submission.json
python tools/build_submission.py
python tools/preflight.py
```

Schur 是研究对照，不在当前单文件导出器支持范围内。若它胜出，导出器会明确拒绝，不能悄悄导出另一个算法。必须先补上并单独验证其提交适配器。

最后才打开保留段，且只评估冻结配置：

```bash
python -m research.backtest --phase holdout --allow-holdout \
  --config configs/submission.json --output outputs/final-holdout
```

不要看完保留段后反复挑参数，再继续称它为未见样本。

### 调仓与交易成本

默认每 20 期重新估计权重、窗口内恒定权重（匹配示例常见的 `X @ w` 评价语义）；`--weight-drift` 切换为窗口内买入持有、权重随收益漂移。老师真实持仓语义未明确，两种均可检查。

`--cost-bps 10` 表示每交易 1 元收取 10 个基点的费用。恒定权重模式会计入窗口内恢复目标权重的交易量，不能只收每 20 期的交易费。费用是按绝对权重变化的资本扣减近似，不是精细订单撮合。默认费用为零，不声称老师也一定按零费用评分。

## 6. 失败与后备路径

主优化 → 有效 HRP 参考权重 → 反波动率 → 最后才使用确定性单资产配置。每条路径都检查最终权重。没有“优化失败就等权”。

非有限值不填成零收益；不足以估计共同协方差时走逐资产风险后备。仅使用最近 252 行，保持原始资产列顺序，不修改输入。稠密风险估计的上限是 2048 个有效资产，它是**成本**护栏而非统计护栏：实测单次 fit 在 256 资产约 0.02 秒、1455 资产约 2.8 秒、2048 资产约 6.4 秒、2900 资产约 12.3 秒。上限取在 2048，是因为标准宽横截面（最宽的常见基准有 1455 列）必须走协方差路径；实测在某真实 1455 资产滚动回测上，放开上限把 Sharpe 从 0.604 提到 1.189、最大回撤从 0.475 降到 0.392、失败折数仍为 0。超过上限才退为 O(TN) 的逐资产风险配置，并记录资源降级。

完全相同资产可能产生由聚类并列决定的不对称权重；全零输入等退化情形无法从数据识别投资优势。若候选输出等权，会转到明确的不同配置规则，不是添加极小噪声。单资产输入只能得到 `[1]`，代码标记“严格禁止等权规则存在歧义”；零行、零资产、非数值或错误维数不属于可保证构造组合的正常输入。

后备成功 ≠ 主优化成功。报告分别记录降级和最终失败。不能保证未见数据、未知依赖版本和任意资源限制下绝无失败。

## 7. 创建你的 GitHub 私有仓库

本次 ChatGPT 的 GitHub 连接能确认账号并读取仓库，但未提供创建/写入动作，当前环境也无已授权 GitHub CLI。因此，**尚未远端建仓或上传**，没有可验证的远端仓库链接。

在自己的电脑上安装 GitHub CLI，使用浏览器登录，不要将访问令牌发到聊天或写进代码：

```bash
gh auth login --hostname github.com
bash tools/publish_github.sh
```

脚本默认创建 **`18047533889/portfolio-game-round1` 私有仓库**。它检查登录账号，只上传固定目录，不上传 `data/`、`.env`、密钥或缓存；不会强推、删除仓库或覆盖不相关的 `origin`。

名称已被其他仓库占用时可明确指定新名字：

```bash
bash tools/publish_github.sh mafs5310-portfolio-round1
```

若 GitHub 拒绝推送 `.github/workflows`，根据 GitHub CLI 的提示补充本人的工作流权限，再重试；不要改为公开仓库绕过。上传完成后仍需看真实 Actions 执行结果，不能把已配置 CI 当成已通过 CI。

## 8. 资料与权属

教师规则以本人提供的截图和 `teacher_reference/` 为准，不自行补出隐藏规则。公开 API 与算法出处见 `docs/sources.md`。

这是私人课程项目，未授权公开再分发。教师原始材料保持其原始权属，不对其重新授予开源许可。
