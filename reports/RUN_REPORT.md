# 本次执行报告

生成时间（UTC）：2026-09-18T17:00 前后（由 `tools/verify_repo.py` 写入 `reports/verification.json`）

## 结论

`submission/portfolio_round1.py` **可以直接提交**。

- `ready_for_teacher_submission`: **true**
- `real_skfolio_preflight`: **PASSED**
- 独立提交文件 SHA256：`13bc2a8a2ec0e46d1e854532f007e8526a8234fc03651fdd8cbd76c56e88cb07`
- 数值核心 SHA256：`67a5a5f6b029089fefa01b9e058671a6ed384612d6b97538e72e466186e02ed5`

## 本次重新执行的检查

| 项目 | 实际结果 | 日志 |
|---|---|---|
| 构建一致性 `build_submission.py --check` | exit 0 | `reports/build-check.txt` |
| 语法 `compileall` | exit 0 | `reports/syntax-check.txt` |
| `pytest` | 95 项：94 通过 / 1 跳过 / **0 失败** | `reports/pytest.txt` |
| 合成压力测试 240 例 | **最终失败 0**，89 例走显式记录的降级路径 | `reports/stress-run.txt` |
| 真实 skfolio 预检 + 老师原版自测 | **PASSED**，`Basic checks passed`，127 个组合、**0 失败** | `reports/preflight.txt` |

## 老师原版自测的关键数字

```
Basic checks passed. Backtest summary:
Annualized Sharpe Ratio                    0.92
Number of Portfolios                        127
Number of Failed Portfolios                   0
Number of Fallback Portfolios                 0
```

## 真实市场回测（`optimize_every = 20`，252 天回看）

属于研究证据，**不是** `verify_repo.py` 的通过条件。用的是 skfolio 自带数据集，不是评分系统的隐藏测试集。

| 数据集 | 资产数 | Sharpe | MaxDD | 年化收益 | 失败窗口 |
|---|---|---|---|---|---|
| sp500 | 20 | 0.9494 | 0.4845 | 0.1442 | 0 / 403 |
| ftse100 | 64 | 1.0228 | 0.4166 | 0.1369 | 0 / 285 |
| factors | 5 | 0.6503 | 0.3935 | 0.1086 | 0 / 100 |
| sp500_index | 1 | 0.5072 | 0.7362 | 0.0932 | 0 / 403 |
| nasdaq | 1455 | 1.1892 | 0.3916 | 0.2798 | 0 / 55 |

nasdaq 一行是放开稠密资产上限后新增的实测：同一份文件、同一口径，上限 256 时为 0.6036 / 0.4746 / 0.1494（与纯反波动率逐位相同），上限 2048 时为 1.1892 / 0.3916 / 0.2798。

## 本次相对上一版的改动

1. **相关矩阵增加 Marchenko–Pastur 特征值去噪**（`rmt_denoise`）。低于 MP 上边缘的特征值方向与抽样噪声不可区分，合并到共同均值后再归一化回单位对角。
2. **锚惩罚由 1.0 调整为 0.25**（`configs/submission.json`）。依据：3 数据集 × 3 时间段共 9 个评价块中最大回撤改善 8/9；同时 5 资产组合最大权重仍被压在 0.535，而纯最小方差会达到 0.833。
3. **稠密风险估计上限由 256 提到 2048**（`MAX_DENSE_ASSETS`）。这是成本护栏：实测单次 fit 在 256 资产约 0.02 秒、1455 资产约 2.8 秒、2048 资产约 6.4 秒、2900 资产约 12.3 秒（树步骤 O(n³)）。旧上限让最宽的标准数据集完全用不上相关结构，放开后 nasdaq 的 Sharpe 由 0.6036 提到 1.1892 且失败折仍为 0。
4. **新增 `tests/test_edge_robustness.py`**（26 项）：低维（n=1/2）合同、MP 去噪的谱性质、稠密上限回归、以及一条会在 skfolio 修复其缺陷时主动失败的前提断言。
5. **新增 `tests/test_build.py::test_frozen_config_is_reachable_and_matches_core_defaults`**：冻结参数与核心默认值不一致时在构建期即报错。
6. **修正 `tests/test_skfolio_integration.py` 的断言口径**：原来假设「适配器所用参数 == `allocate()` 的默认值」，现改为与 `configs/submission.json` 的冻结值比较。

`reports/github-publication.json` / `.txt` 仍保留上一次的 `BLOCKED_BY_PLATFORM_SAFETY_CHECK` 记录。**那份归因是错的**：真实原因是 HTTPS OAuth token 缺少 `workflow` scope，无法推送 `.github/workflows/tests.yml`。改用 SSH 后推送成功，远端 main 已包含完整代码。保留旧记录是为了可追溯，不作为当前状态。

## 未完成的检查

- `tools/publish_github.sh` 面向私有仓库设计，不适用于用户当前提供的公开目标仓库。
- 隐藏测试集不可见，本报告不宣称对其的表现。
