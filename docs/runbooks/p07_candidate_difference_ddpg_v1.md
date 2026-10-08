# 差分目标的匹配 DDPG 验证

目录 `E:\Prediction_RL_worktrees\response-difference-v1`，环境 `pytorch`，
分支 `codex/p05-response-difference`。旧 `E:\Prediction_RL` 仅作只读来源。
独立完整配置：`configs/development/p07_candidate_difference_120k_v1.json`。
原环境/奖励：`RL-MPC-LaneMerging-master/configs/train_default_1.json`。

## 冻结设计

只新增差分 B3：训练种子 0/1/2，各从头训练 120k，总请求 360k 步。
保留 20k/40k/60k 诊断节点，120k 为唯一主终点，完整 episode 边界最多
超出 499 步并记录实际步数。CPU 一线程、训练串行、lr=.0002、warmup=5000、
batch=100、原 2M scheduler、原连续 jerk/reward/终止语义均不变。

评价沿用 Gym＋20 秒预热，场景 200–219，新进程首回合，无探索和更新。
新增 3×4×20＝240 回合；只读复用 P7n 四组 960 回合。仍只有 20 个不同
交通场景，不是 1200 个独立样本或新确认集。主要比较差分 B3 − 原 B3，
同时报告相对 B2、B0 的差异。不加入 Shield/MPC、commitment、Author50 或泛化。

新预测器、验证和独立校准分别在 `artifacts/prediction_difference/delta_v1/`
和 `artifacts/difference_calibration/delta_cal_v1/`。不要重复已有 prepare/run。
149 维在线特征继续只用预测均值，不使用 bands；宽校准区间不是精确安全保证。
旧 loader 拒绝新版本，显式 difference loader 严格绑定三成员、配置和权重。
旧 B3 策略权重不允许作为新方法正式结果，也不作为训练初始化。

运行依赖 st_cy.cp310-win_amd64.pyd 从只读来源复制到新工作树，未重新编译，
SHA256：498a57f2096e905bb1151eb37dfde41cfb1b76668c343c16e55ae63fcebc76b3。
此二进制被 Git 忽略，不发布；源码和 SUMO/ALL 身份仍检查。

Git 工作树可能将同一上游配置检出为 CRLF，而历史运行使用 LF。复用校验
将历史字节哈希绑定到当时 Git blob，逐字验证新旧配置仅有 LF/CRLF 差异，
连额外空白也不接受。新请求同时保存历史/当前字节哈希；不覆盖历史报告，
新策略训练和评价仍严格使用并绑定当前配置字节哈希。

## 有界 smoke 与准备

开发方先做一个 64 请求步训练 job 和三次场景 200 冻结评价；warmup/batch=8
只用于工程验收，不是学习效果证据。负驾驶结果正常记录。

```powershell
python -B tools/train_difference_ddpg.py audit --calibration-request artifacts/difference_calibration/delta_cal_v1/request.json --run-id delta_smoke_v1
python -B tools/train_difference_ddpg.py prepare --calibration-request artifacts/difference_calibration/delta_cal_v1/request.json --engineering-audit artifacts/difference_ddpg/delta_smoke_v1/audit.json --run-id delta_120k_v1
```

若已准备，不重复执行同 run ID，直接使用请求。运行期间源代码、配置、环境
冻结且工作树干净；后续纯文档提交不改变 source hash 时可继续消费。

## 用户手动训练、评价、聚合

2026-10-08 已完成上述有界 smoke 和 prepare，不必重复执行它们。
正式请求 hash：`e2c86b9f2310b2d9a49a6fe00c5c284a7071102ac153ad484f2b7f9c95e1d372`。
工程验收见 `reports/p06_difference_connection_acceptance_20261008.md`。
当前长训练/正式评价尚未启动。

```powershell
cd E:\Prediction_RL_worktrees\response-difference-v1
$differenceRequest = 'artifacts/difference_ddpg/delta_120k_v1/request.json'
$differenceHash = (Get-Content artifacts/difference_ddpg/delta_120k_v1/preparation.json -Raw | ConvertFrom-Json).request_hash
python -B tools/train_difference_ddpg.py run --request $differenceRequest --confirm-request-hash $differenceHash --stage train --resume
python -B tools/train_difference_ddpg.py run --request $differenceRequest --confirm-request-hash $differenceHash --stage evaluate --resume
python -B tools/train_difference_ddpg.py aggregate --request $differenceRequest --confirm-request-hash $differenceHash
```

按顺序执行。resume 只复用 sealed COMPLETE 整个 job，不从评价权重恢复
optimizer/replay；失败/中断目录保留，拒绝自动重试。聚合只执行一次，不覆盖。
不删除训练种子，不选较好的中间节点，不自动延长预算。

## 输出与判断

`artifacts/difference_ddpg/delta_120k_v1/` 下保存 train 节点权重、实际步数、
原 actor/critic loss、replay 和 episode 日志；evaluate 保存动作、状态和原始
终止事件。training_curves.json 是轻量日志索引，不是独立验证曲线。
development_curves.json、aggregate.json 和 aggregate_receipt.json 保存四节点
五方法表及逐场景/训练种子配对；旧巨大 replay 不复制。

旧 B3 开发到达率已接近上限，同时审查原回报、碰撞/停车、任务期限、
动作饱和、jerk 和成功回合完成时间。全回合时长混合失败结局，不能单独
解释成功合流效率。离线改善不能代替闭环收益；完成后作明确保留/放弃决策。
