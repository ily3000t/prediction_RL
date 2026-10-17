# RCCI02-A 离线 adapter 实现验收（2026-10-17）

## 实际完成

独立工作树 `E:/Prediction_RL_RCCI`，分支
`codex/road-conflict-consequence-interface-v1`；本轮起点 `16faa29`。
本轮只新增文件，不修改已有 Raw、预测器、DDPG、环境、配置或历史产物。

- `features/road_consequence_geometry.py`：当前合法路线投影、有限 merge-node
  凸控制体积、road-aligned front-bumper footprint、占用窗口与截尾。
- `features/road_consequences.py`：mean-track-first 的 149 维独立离线纯函数，
  front/rear route-gap 池化、跨来向占用描述、有效性 masks、逐候选诊断。
- 新测试、详细字段/尺度/掩码契约及运行说明。

未注册 online provider。维度相同不代表 Raw 权重可直接用于 Relation。
没有运行 predictor inference、SUMO、新数据收集、ensemble/DDPG 训练或闭环评价；
没有打开 calibration/test，没有新优化器/仿真种子。

## 实际测试

环境：`E:/Programs/EnvAnaconda3/envs/pytorch/python.exe`，Python 3.10.16、
PyTorch 2.5.1、NumPy 2.2.6（与 RCCI01 同一本机环境，无安装/升级）。
测试是开发工作树上的合成/离线回归，不是正式实验。

1. 新测试首次运行：21 passed、2 failed。
   失败暴露最小 gap 数值平局的选择时刻会受浮点舍入/共同刚体变换影响。
   修正为 1e-6 m 纯数值平局、最早时刻、车辆 ID；没有更改安全/方法效果阈值。
   原临时目录 `artifacts/tests/rcci02_unit_v1` 保留。
2. 修正后的首批新测试：23 passed；目录 `rcci02_unit_v2`。
3. 补充掩码/窗口/版本/候选独立性测试后，完整选定离线回归：
   **223 passed、4 deselected，14.02 s**；其中新增 RCCI02 为 28 项。
   目录 `artifacts/tests/rcci02_reg_v1`。

实际回归命令：

```powershell
python -B -m pytest tests/test_road_consequences.py tests/test_rcci_feasibility.py tests/test_probe_features.py tests/test_dataset_contract.py tests/test_candidate_ranking.py tests/test_task_relevance.py tests/test_response_collection.py tests/test_masked_training.py tests/test_all_ddpg_features.py tests/test_training_contract.py -q -p no:cacheprovider -k 'not test_actual_upstream_reward_done_and_cleanup' --basetemp E:/Prediction_RL_RCCI/artifacts/tests/rcci02_reg_v1
```

四项未运行是原 parametrized native import 检查，本工作树仍无被 Git 忽略的
`st_cy` 扩展。没有关闭 Cython、修改依赖或放宽旧测试来绕过；不能称全项目全部通过。

覆盖：无车/遗漏、稳定 IDs、负 route gap、closing speed、26 点时域、窗口插值、
左右截尾/未到达/已通过、有限区与车辆长宽、端点/离路/歧义/倒退/负速度、
候选批量与逐个一致、future poisoning、后处理共同刚体变换和旧 Raw 相关回归。

## 科研解释与停止点

这里只证明实现按所冻结的近似契约运行，不证明方法有效、占用等于碰撞，或模型
能给出可信安全概率。finite control volume 是 merge node 的凸包，不是精确
车身冲突区域；route gap 在连接道路上也不是即时欧氏净空。

0.5 m 投影支持半径未经真实冻结预测覆盖验收；完整池 mask 可能使很多数值不可用，
这应被如实测量，而不是删除难算车辆。有限预测无法投影与时域截尾明确区分。
三成员仍先平均轨迹，没有加入路线 B 的成员级后果聚合。

RCCI02 的契约、adapter 与合成单元部分完成；**真实 train/validation 上的冻结
预测覆盖、几何偏差、窗口误差及 B2/B3 候选分解尚未完成**。
下一步应另建不可变、有界、只读离线预测审计，先核查上述量，不能直接跳过进入
RCCI03 原生接入或 RCCI04 正式训练，也不能重跑/覆盖 RCCI01 历史请求。

当前结果对创新性的评价不变：这是有合理动机的接口机制研究，尚未建立性能增益；
不能把加入地图/完整时域/占用特征本身写成首次提出或安全保证。

## Git 边界

只在当前 `codex/` 分支作原子本地提交，不合并、不打 tag、不推送。
`E:/Prediction_RL` 在修改后检查仍为干净 `main`；旧项目及两个归档研究工作树
均未修改。历史 RCCI01 请求/报告不覆盖、不松绑 source guards。
权重、测试临时目录与原始产物均不加入 Git。
