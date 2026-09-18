> **本次状态更新：** 请先阅读 [reports/RUN_REPORT.md](reports/RUN_REPORT.md)。2026-09-18 重新执行本地检查：58 passed、2 skipped；240 组压力测试最终失败 0；252/20 合成回测 28 个窗口失败 0。真实 skfolio 验收与真实市场回测未完成。GitHub main 目前仅 README.md，完整代码上传被平台安全检查拦截，Actions 未启动。下文是原项目使用说明，不代表本次已经完成其全部步骤。原 `publish_github.sh` 以私有仓库为保护前提，不适用于用户现在提供的公开目标仓库。

# Portfolio Game Round 1｜组合优化作业与研究工具

MAFS5310 第一轮。只使用老师提供的资产收益率，不使用额外因子、外部行情、预训练模型或虚构现金资产。

> **本次交付状态：代码和数值测试已完成，但真实 skfolio 验收尚未完成；GitHub 远端尚未创建/上传。**
> 最新实际运行记录见 `reports/verification.json` 和 `reports/ACCEPTANCE.md`。
> 不能把数值测试通过当作老师自测通过，也不能把合成数据测试当作市场回测成绩。

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

252 期历史收益 → Ledoit–Wolf 收缩协方差 → 近期波动/长期相关性混合 → 树结构 HRP 参考配置 → 正则化最小方差。

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

非有限值不填成零收益；不足以估计共同协方差时走逐资产风险后备。仅使用最近 252 行，保持原始资产列顺序，不修改输入。超过 256 个有效资产时不创建大规模稠密协方差矩阵，记录资源降级并采用逐资产风险配置。

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
