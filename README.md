# MAFS5310 · Portfolio Game Round 1

本仓库区分教师提交代码与内部研究、回测、调参、测试代码。

## 给老师提交哪个文件？

**只提交 [`submission/portfolio_round1.py`](submission/portfolio_round1.py)。** 不提交整个仓库、配置、数据、研究程序或老师的示例。

该文件自包含，类名为 `CVXPYPortfolio`，继承真实 `skfolio.optimization.BaseOptimization`。输入是历史资产收益率，输出是一维 `weights_`；只做多、满仓、不加杠杆，最终检查非等权。

## 当前状态

正在导入项目并进行真实依赖验收。先前本地运行只完成数值测试，不等于老师的自测已通过。最终以本仓库 Actions 对应提交的实际结果和 `reports/RUN_REPORT.md` 为准。

## 算法

252 期收益率 → Ledoit–Wolf 收缩协方差 → 近期波动调整 → HRP 参考权重 → 正则化最小方差凸二次规划。使用有迭代上限和最优性误差检查的数值求解；失败时保留明确的后备配置及诊断。默认半衰期 63、近期风险混合比例 0.25、参考正则化 1.0；这些是预先冻结的初始参数，不是市场回测选出的最优值。

## 目录

| 路径 | 用途 |
|---|---|
| `submission/portfolio_round1.py` | 唯一教师提交文件 |
| `src/portfolio_game/core.py` | 内部维护的数值核心 |
| `configs/submission.json` | 冻结参数 |
| `research/` | 数据处理、滚动回测、比较与调参；不交老师 |
| `teacher_reference/` | 用户提供的原始老师样例和自测，保持原样 |
| `tests/` | 数值、隔离、真实 skfolio 和独立求解器测试 |
| `tools/` | 构建、自测、压力测试及上传工具 |
| `reports/` | 验收记录；云端原始日志亦见 Actions artifacts |
| `docs/` | 算法、边界和参考文献 |

## 本地运行

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python tools/verify_repo.py --require-integration
```

`--require-integration` 缺少真实依赖或老师自测未通过时必须失败，不用模拟库冒充成功。

验证期 252/20 回测：

```bash
PYTHONPATH=src:. python -m research.backtest --phase validation
PYTHONPATH=src:. python -m research.backtest --phase validation --weight-drift --output outputs/drift
```

调参（仅开发与验证期，不自动打开最终保留期，不自动修改教师文件）：

```bash
PYTHONPATH=src:. python -m research.tune --top-k 3
```

更改参数后必须显式重新生成并验收：

```bash
python tools/build_submission.py --config configs/submission.json
python tools/verify_repo.py --require-integration
```

## 边界

- 真实测试不等于隐藏测试成功保证；当前公开数据不代表完整隐藏测试集。
- 老师原版自测为 252/63；研究另测试 252/20。固定目标权重与窗口内买入持有的漂移模式分别记录。
- 零交易费用、零无风险利率是研究假设，非已确认评分设置。收益率、回撤等结果只能在这些假设下解释。
- 禁止等权的容差及单资产输入的处理仍需老师明确；零资产无可行投资组合。
- 仓库不上传行情原始数据、密钥或本地环境。当前仓库由用户创建为公开仓库，未修改可见性。
