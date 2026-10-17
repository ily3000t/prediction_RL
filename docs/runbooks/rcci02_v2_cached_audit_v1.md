# RCCI v2：保存预测上的零前向重算

工作目录 `E:/Prediction_RL_RCCI`，Python 使用 pytorch 环境。
契约先于实现与重算冻结于 `1daa643`：
[v2 定义](../design/rcci02_adapter_contract_v2.md)。
配置 `configs/development/rcci02_v2_cached_audit_v1.json`，独立可读，无 extends。

## 范围

- 仅读同一六根的已保存 ensemble mean 与 ego rollout。
- 模型权重仅核验既有 SHA，不 `torch.load`，模型前向=0、SUMO=0、训练=0。
- 不解析新的原生 future target，不向 v2 传入终止/存在真值 mask；只使用当前
  history、地图、缓存预测。既有输入文件可做字节哈希核验。
- 不扩大 cohort，不改 seeds、阈值、actor slots 或任何旧源码。
- v1 特征必须精确复现，完整 5 s 支持 flag 必须与 v1 完全一致；不将部分支持
  重新命名为完整覆盖。仅有 partial 描述的候选仍明确保留未知后缀。

## 执行

工作树必须干净，run ID 必须不存在；不覆盖既有父请求/报告或失败目录。

```powershell
python -B tools/audit_rcci_v2_cached.py prepare --run-id rcci02_v2_bnd_v1
```

复制打印的实际 hash：

```powershell
python -B tools/audit_rcci_v2_cached.py run --request artifacts/rcci02_v2/rcci02_v2_bnd_v1/request.json --confirm-request-hash HASH_FROM_PREPARE
```

来源链：冻结数据/模型 → 六根预测审计 → 倒退诊断保存的 forecasts → 本次 v2 重算。
本次新请求分别绑定父 artifacts、71 个既有源文件、旧 source 内容、新完整 source、
Git/environment/config、固定设计文档、v2 字段契约与地图 SHA。
只允许新增 v2 实现；旧 guards 不修改，也不执行已完成的旧请求。

## 输出与解释

`artifacts/rcci02_v2/<run_id>/`：request、preparation、started、roots、report、receipt。
raw 预测仍只保存在父目录，避免复制；roots 保存新特征、支持前缀和 pair 诊断。
未成功运行看不到 complete；负结果正常 complete，损坏身份/NaN/重现失败显式 failed。

分别检查以下结果，不能混成一个通过率：

1. 完整 5 s following/cross 支持数量：必须保持原值。
2. supported-prefix front/rear minimum_available、支持时域 T 与未知后缀数量。
3. prefix 内 overlap_available 与正 overlap 数：其定义不同于 v1 需两车进入后才
   可计算的完整窗口描述，不是同一预测指标变好。
4. root-only 时域、root 时刻即为最小 gap/已进入的数量，防止把当前状态当未来预测。
5. cross actor 选中是否变化：新的前缀与冻结池化语义可能更换描述对象，不等于
   实际执行动作改变或策略收益。

六根已经用于设计诊断，不是独立验证；两个 episode 不能由 actor/candidate/time
扩充成大样本。该报告不比较原奖励、到达/碰撞，也不证明预测准确性或方法效果。

## 测试与停止点

合成测试命令（basetemp 换用尚未存在的短目录）：

```powershell
python -B -m pytest tests/test_road_consequences_v2.py tests/test_rcci_v2_cached_audit.py -q -p no:cacheprovider --basetemp artifacts/tests/rcci02_v2_user_01
```

未注册在线 provider；不修改旧训练命令，不用 Raw/v1 权重直接解释 v2 输入。
本轮之后审阅字段支持，再冻结一次更广的 train/validation 离线评价范围；如需
RCCI03 原生 smoke 或正式 DDPG，须另行批准。此处不提供尚未实现的正式训练 CLI。
