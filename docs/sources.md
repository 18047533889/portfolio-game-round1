# 来源与适用边界

## 教师原始材料

用户提供的两张截图规定 Round 1：禁止等权、禁止卖空、leverage=1、lookback=252、optimize_every=20；评分百分位权重分别为夏普10%、最大回撤10%、年化收益10%、运行失败率70%。

`teacher_reference/self_test.py` 与 `example_portfolio.py` 来自用户上传，逐字节保留，哈希见同目录 SHA256.json。原版自测窗口实际为 252/63，只检查基本格式，不检查全部本轮规则。不得把样例的等权后备移入正式提交版本。

未知：教师精确依赖版本、隐藏市场/资产数、等权容差、指标的精确年化口径、无风险利率、交易成本、是否允许现金余额、窗口内持仓漂移语义。项目按可见样例的非负、满仓约束实现，不把未知设置写成事实。

## 公开主要文档（2026-09-18核查）

- skfolio BaseOptimization： https://skfolio.org/generated/skfolio.optimization.BaseOptimization.html
- skfolio SchurComplementary： https://skfolio.org/generated/skfolio.optimization.SchurComplementary.html
- skfolio CovarianceDistance： https://skfolio.org/generated/skfolio.distance.CovarianceDistance.html
- skfolio BaseCovariance： https://skfolio.org/generated/skfolio.moments.BaseCovariance.html
- skfolio EmpiricalPrior： https://skfolio.org/generated/skfolio.prior.EmpiricalPrior.html
- 公开样本入口： https://skfolio.org/generated/skfolio.datasets.load_sp500_dataset.html
- CVXPY 求解器文档： https://www.cvxpy.org/tutorial/solvers/index.html
- GitHub CLI 创建仓库： https://cli.github.com/manual/gh_repo_create

当前集成依赖参照官方仓库 2026-09-16 的 v1.2.8 标记。并未据此声称已在该版本上通过运行测试；以 reports/verification.json 为准。

## 文献思想与本实现的区别

Ledoit–Wolf 收缩与 HRP 是成熟方法，本项目没有把它们称作“2026 年新算法”。前轮讨论的新论文用于设计取舍，本次交付没有复现 Wasserstein DRO、神经协方差或拓扑风险平价，也没有声称这些方法在隐藏数据上孰优。

主方案是经用户批准的工程组合设计；Schur 使用 skfolio 的真实研究实现，不在其不可用时悄悄替换成 HRP。由于未完成真实市场回测，目前没有文献外推的收益数字可报告。
