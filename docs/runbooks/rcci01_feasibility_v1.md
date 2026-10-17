# RCCI01 离线审计运行说明

工作目录 E:\Prediction_RL_RCCI，Python 使用 pytorch 环境。
只读资产目录 E:\Prediction_RL；不需要复制模型、原始数据或编译 st_cy。
配置：configs/development/rcci01_feasibility_v1.json。
契约与范围：docs/design/rcci01_feasibility_contract_v1.md。

2026-10-17 已完成 rcci01_bnd_v2 与 rcci01_full_v1；不要重复这些 run ID。
实际结果及限制见 [验收记录](../../reports/rcci01_feasibility_acceptance_20261017.md)。
下方命令说明完整流程，不代表还需要再跑一次已完成审计。

## 有界开发审计

工作树须干净，run ID 不存在时才运行：

```powershell
python -B tools/audit_rcci_feasibility.py audit --baseline-root E:/Prediction_RL --run-id rcci01_bnd_v1
```

最多 train/validation 各前三根，每根五个已有候选分支。
固定前缀不是方法效果样本，不据此推断整个数据集的覆盖率。
不启动任何 SUMO、预测器前向或训练；两个模型仅检查 SHA。

## 完整 train/validation 审计

prepare 会核验所用输入文件，不运行模型或仿真。只准备一次：

```powershell
python -B tools/audit_rcci_feasibility.py prepare --baseline-root E:/Prediction_RL --scope full --run-id rcci01_full_v1
```

复制打印的 confirm_request_hash，再运行：

```powershell
python -B tools/audit_rcci_feasibility.py run --request artifacts/rcci01/rcci01_full_v1/request.json --confirm-request-hash HASH_FROM_PREPARE
```

共 713 train + 177 validation 根，不排除空监督根或按结果选样。
磁盘读入现有 history/labels/c0–c4，可能比 bounded 审计耗时；不是长时间
SUMO 评价或新的预测/策略训练。运行期间保持基线、工作树源码和配置冻结。

## 产物与解释

artifacts/rcci01/<run_id>/ 下：

- request.json：实际 Git/source/config/environment、只读资产路径、输入 SHA、
  原始 collection jobs（含其 seeds）和根族清单；无新 optimizer seed。
- preparation.json：准备时的不可变状态，不是最终运行状态。
- started.json、report.json、receipt.json：状态、限定审计结果、输出哈希。
- roots.json：逐根及逐候选的小型审计统计；不复制原始 future trajectories。

report 的 known-lane projection 使用离线 lane 标签，不是预测投影准确率。
lane 转换不等于冲突事件；heading 缺失不允许制造原生 footprint 真值。
事件提取、线上 adapter、模型 runtime 验收和闭环方法效应仍未发布。
负面几何结果正常 status=complete；损坏哈希/NaN/时间或ID错位会显式失败。

失败目录保留，不覆盖或自动重试；修复工程错误后使用新 run ID。
不改 seed、阈值或删样本让结果通过。RCCI02 须在审阅本阶段后另行冻结。

## 工作树构建限制

本工作树未包含被 Git 忽略的 st_cy 编译扩展。RCCI01 不导入原生控制栈，
不依赖它。涉及 upstream_session 的既有原生导入测试会因缺少 st_cy 失败；
不能据此宣称整套上游 runtime 回归通过，也不通过关闭 USE_CYTHON 绕过。
原生构建及运行兼容检查留在 RCCI03，届时单独记录版本和构建产物。
