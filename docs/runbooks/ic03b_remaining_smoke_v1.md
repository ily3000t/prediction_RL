# IC03b 修复版：只执行剩余三组

用户在旧运行完成首组两个原生回合、135 控制步后，明确批准继续剩余三组、最多六个新回合、零优化更新。授权记录为 `provenance/ic03b_remaining_smoke_authorization.json`；独立、无 extends 配置为 `configs/development/ic03b_remaining_smoke_v1.json`。

只运行 B2-Curriculum、B3-Mix、B3-Curriculum，每组一个新初始化的持久原 ALL DDPG learner，两个原生回合。保留初始化 seed 0、sampler seed 17003、原 5,000 warmup、uniform replay、continuous jerk、reward/termination/bootstrap、冻结三成员 predictors、fresh Gym20 入口与阻塞推理。零 optimizer 更新，不加载旧 policy，不启动正式训练，不重新采反事实。

新请求上限：六会话、3,000 控制步、3,612 raw ticks、3,006 特征调用。旧消耗是两会话、135 控制步、337 ticks、135 特征调用，不能报告为零。B2-Mix 不在新 CLI worker choices 或 jobs 中，不得重复运行。每组沿用旧两回合 sampler contract，该低层模板中的原父预算不作为新执行权限；新父协调器严格使用六会话配置。

只读复核整个旧失败目录的 21-file inventory 指纹、原请求与科学来源、旧 receipt 除最终日志提示外所有字节，并重新核算封存 raw events 的 reward/tick/history。旧 request 仍 failed，旧 complete 仍无效，不修改/重封存原产物，不事后重建旧内存 tensor 或 RNG；旧 child 已记录的检查与新六回合的有效收据在 coverage_review 中分别标明来源。

通过显式函数参数复用已修复的 worker/父协调器，不 monkey-patch 原 loader，不绕过旧 immutable gate；没有修改原 scientific modules。父进程等待成功退出、关闭 worker.log 后才 seal 全目录，日志仍进入 inventory。

在干净课程工作树的 pytorch 环境中：

```powershell
python -B tools/smoke_remaining_curriculum.py prepare --baseline-root E:\Prediction_RL
python -B tools/smoke_remaining_curriculum.py run --request artifacts/ic/ic03b_remaining_v1/request.json --confirm-request-hash <prepare打印的hash>
```

固定新 run ID，不覆盖、不自动 retry/incomplete resume。失败时停止并保留已消耗预算；不能再次运行 prepare 或通过换路径获得又六回合。真实输出仅在该工作树 `artifacts/ic/ic03b_remaining_v1/`，包含新 request、逐组 native/replay/summary、完整收据、report 与独立 coverage_review；旧目录只读。

工程 gate 不要求未训练策略表现好，合法碰撞/到达/期限均正常保留。新六回合完整通过也只证明工程接入，不证明课程收益，不授权四组×三个 seed×120k。下一步是 IC04 正式请求冻结与人工审查，由用户启动耗时训练。

## 实现的离线测试

19 项新增测试覆盖范围不可扩张、旧失败字节不变与独立 raw-event 重算、原科学来源不可忽略、仅三组任务、收据关闭后封存、失败停止且禁止再执行、累计预算包含旧消耗、六会话而非旧八会话 gate，以及每组 sampler 只能显式复用原契约。定向回归 77 passed（8.84 秒），全项目离线回归 1,295 passed（36.83 秒、两个原作者 torch.load FutureWarning）。这些测试没有启动新的 SUMO 或真实 predictor 推理，不代表六回合实际验收已完成。

## 2026-10-11 真实执行已完成

实现提交 `83440d967f64269a18a42d503aa5e8fd20ec6b09` 上冻结并执行，request hash 为 `3f5f910adbda5ed1242e19d7630fe85d03a3d2e1d507f242f076bb95e9d98ec4`。新六回合完成：1,147 控制步、1,753 raw ticks、1,147 特征调用、零优化更新；三组收据及独立 raw/replay 复核通过，engineering_gate=true。旧加新八会话共 1,282 控制步/2,090 ticks；旧失败/失效收据仍不变。

详见 `reports/ic03b_remaining_smoke_acceptance.md`。上面的 prepare/run 命令现为不可重跑的历史命令。下一步只进入 IC04 请求冻结与人工审查，formal_training_authorized/training_ready 仍 false；不得把未训练 smoke 到达率视为课程收益。
