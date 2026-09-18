# 本次执行报告：上传受阻，数值测试已重新运行

生成时间（UTC）：2026-09-18T13:16:48.274310+00:00

## GitHub 实际状态

目标：`18047533889/portfolio-game-round1`（用户创建的公开仓库，未改变可见性）。

已成功创建 `README.md`，main 提交为 `2eca5f381569f5c0a7a3b0d5e39abd0f3eb20d15`。随后已回读 main 文件列表，**当前只有 README.md，完整代码没有上传到 main**。

数值核心及基础配置创建过未提交的 Git tree 对象；它不等于可见的代码提交。研究目录批量上传被平台安全检查拦截，消息为：

> 因 OpenAI 无法确定请求的安全状态，已拦截此工具调用。

连接读取到 push 权限为 true，因此不能将这个阻塞说成“仓库没有写入权限”。GitHub Actions 尚未启动。

## 本次重新执行的检查

| 项目 | 实际结果 |
|---|---|
| `python -m pytest -q` | 58 passed，2 skipped，0 failures |
| 提交文件与核心、配置一致性 | 通过 |
| Python 语法编译检查 | 通过 |
| 合成压力测试 | 240 组，最终无效权重 0 组 |
| 压力测试降级/非等权后备 | 98 组 |
| 压力测试主优化 | 尝试 162 次，收敛 150 次，未收敛后转后备 12 次 |
| 252/20 合成滚动流程 | 28 个窗口，最终失败 0，降级 0 |
| 真实 skfolio 集成 | 未执行：缺少 skfolio |
| 独立 CVXPY 对照 | 未执行：缺少 cvxpy |
| 老师原版 self_test.py | 未执行 |
| 真实市场回测/调参 | 未执行 |
| 严格验收退出码 | 2，未通过完整验收 |

这两项 skip 是缺失依赖引起的测试模块跳过，不是相应集成用例已经通过。

## 环境与依赖

Python：3.13.5 (main, Jul 15 2026, 20:25:40) [GCC 14.2.0]

```json
{
  "numpy": "2.3.5",
  "scipy": "1.17.0",
  "scikit-learn": "1.8.0",
  "pandas": "2.2.3",
  "pytest": "9.0.2",
  "skfolio": null,
  "cvxpy": null,
  "cvxpy-base": null
}
```

本次安装命令：`python -m pip install -r requirements-dev.txt --retries 0 --timeout 10`，退出码 1。输出包含 `No matching distribution found for skfolio==1.2.8`；另一次网络探测报告 `Could not resolve host: pypi.org`。这是当前执行环境中的失败，不足以证明 PyPI 上不存在该版本。

严格验收命令：`python tools/verify_repo.py --require-integration`，退出码 2。没有伪造依赖，没有把缺少依赖的检查改成通过。

## 教师文件

唯一提交文件：`submission/portfolio_round1.py`。

SHA256：`4504629640d1fa5c9050cf0e5002f5891ad164e9453e04e1ee8a28a587e39360`。

本次没有更换算法、没有调优参数，文件仍是事前冻结参数的候选版本。**还不能标记为完成老师环境验收的版本。**

## 原始日志

- `reports/pytest.txt`、`reports/pytest.xml`
- `reports/stress.json`、`reports/stress-run.txt`
- `reports/synthetic-current.txt`
- `reports/dependency-install.txt`
- `reports/preflight.txt`、`reports/engineering-run.txt`
- `reports/github-publication.json`

合成数据仅检查工程流程，不是投资业绩证据，不将其夏普或年化收益用作策略效果结论。隐藏测试、评分器容差、教师实际依赖版本仍未验证。
