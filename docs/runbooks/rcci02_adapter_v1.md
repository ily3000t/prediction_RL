# RCCI02-A 离线 adapter 检查

工作目录：`E:\Prediction_RL_RCCI`。不需要复制权重、启动 SUMO 或编译扩展。
详细语义见 [接口契约](../design/rcci02_adapter_contract_v1.md)。

## 纯离线单元测试

以下 basetemp 必须换成尚未存在的短目录，避免 pytest 清理历史测试输出：

```powershell
python -B -m pytest tests/test_road_consequences.py -q -p no:cacheprovider --basetemp artifacts/tests/rcci02_user_01
```

库入口：

```python
from prediction_rl.features.road_consequence_geometry import RoadConsequenceMap
from prediction_rl.features.road_consequences import contract, summarize_road_consequences

road = RoadConsequenceMap("RL-MPC-LaneMerging-master/merge.net.xml")
schema = contract()
# history: 仅已有观测；neighbors: 原 ensemble mean；ego: 原固定五探针 rollout。
# result = summarize_road_consequences(history, neighbors, ego, road)
# result["features"] 为 149 维（五候选），result["candidates"] 为逐候选诊断。
```

上述是数据接口说明，不是已经存在的实验 CLI。当前不修改旧 DDPG 启动命令，
不把旧 Raw PPO/DDPG 权重拼到新语义后直接评价，也不要求重训预测器。

## 停止点

纯函数实现阶段只完成契约、实现与离线合成测试。后续已完成六根真实冻结预测的
有界审计，见 [结果](../../reports/rcci02_frozen_audit_acceptance_20261017.md)与
[审计命令](rcci02_frozen_predictor_audit_v1.md)。这不是全数据覆盖验收，且发现完整
池掩码可用性及 margin 支持问题；不得直接跨到正式策略训练或以测试通过代替方法效果。
