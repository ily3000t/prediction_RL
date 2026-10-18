# RCCI04c：定期checkpoint序列化修复与新请求交接

## 原因与保留证据

原请求 `artifacts/rcci04c/matched_v1/request.json`，确认hash
`8dfa87d77539dd7e6167127aa8a0299b8cd1a680bc62b979f58fec6f0b076b20`，
在 `rcci_ordinary_s0` 的第200次优化更新保存policy时失败：

```text
AttributeError: Can't pickle local object 'Runtime.factory.<locals>.build.<locals>.count'
```

这是RCCI入口新增审计hook的工程错误。ALL原实现每200次更新调用
`torch.save(model, ...)`，会连同model上的hook一起序列化；旧hook为局部闭包，无法pickle。
Gym/NumPy提示、模型方法效果、奖励、种子及SUMO不是这次退出的直接原因。

最后一个完整episode日志为frame5083；失败loss记录为training_steps5148、sample_call200。
只有该job启动，没有complete receipt或20k节点snapshot，未启动后续五个训练或正式评价。
此前两组入口短训练只有45/58次更新，没有覆盖第200次定期保存；不能用其通过结果
声称完整长训练路径已验证。此次补齐专门的真实定期保存回归。

旧目录所有内容保留，不覆盖、不删除、不把不完整policy/q日志文件当可续训checkpoint。
核对的原证据文件SHA256：

- request.json（文件字节）：`b6c253565ffa2419d8d8aa7316cd5eed84dc2b7d7a9479f438fc035b1e01f957`
- invocations/89983c4a0c/rcci_ordinary_s0.log：`e3193f08d2b78bbcd81a081f7bea61c781ca7565a1d8f084ff428193b0204ced`
- train/rcci_ordinary_s0/failure.json：`6406e65a1ac5b4f336b94c58b10dad05cc554822185fde1e935bed8d96c8d3e3`

## 修复边界

源码提交：`0230f8510634da2d43b6cbb2c19fae2d482fa5aa`。

- 新增可导入、可pickle的 `PolicyForwardCounter`，只持有整数调用数及可选cap，
  不捕获Runtime、provider、文件流或局部函数，不修改任何输入/输出或随机数。
- policy与target-policy仍共享同一计数器，原计数口径和smoke上限不变。
- 不关闭或更改ALL定期保存，不修改安装包、原DDPG函数、优化器、replay或权重contract。
- 子进程退出时，父进程直接给出job、exit code、日志绝对路径及半成品不可续训说明。
- 不放宽旧请求源码校验。修复后必须使用新源码、新入口验收和新不可变请求。

未修改正式配置、actor/road接口、模型结构、预测器、奖励、seed、训练步数或评价集。
E:/Prediction_RL保持只读干净75051acc；E:/WCDT_ACCVP未修改。

## 实际验证

新增测试覆盖两种RCCI arm的实际Runtime hook注册、原ALL第200次定期保存，
不缩短保存频率。policy/q文件均实际生成，target也验证可序列化，
保存的policy/target在新Python进程加载成功。

同一输入下，对比不加hook与修复后hook：动作、Torch/NumPy RNG、policy/q参数、
target参数、Adam状态和学习率配置精确一致。计数包括行为、policy更新和target调用。
cap边界与pickle往返单独测试。不使用SUMO或额外正式训练验证这一保存边界。

- `test_rcci_matched_policy.py`：12 passed（包含在下一组，不重复计数）。
- RCCI/DDPG/桥接等14个选定测试文件：224 passed，34.89s。
- `test_road_consequence_geometry_cached.py`：9 passed，3.22s。
- 合计233项选定回归通过；不是声称仓库全部测试。
- `git diff --check`通过。

在同一修复源码上重新运行原冻结的有界SUMO入口，不增加其预算：

- 目录：`artifacts/rcci04c/entry_savefix_v1`。
- request hash：`0380cd6ab96786326982c4721a8a8bb5ae8e98b87b614ec23a8efcbb99e75321`。
- aggregate文件SHA256：`20e7b6950664b13e78dd22815864474ab95805be97ee613d95c3f56babbadc5c`。
- complete、engineering_complete=true；6个job完整receipt；无ST调用。
- 实际6回合、965控制步、1572 simulation steps、965 ensemble/2895 member forwards、
  1177 policy forwards，全部在冻结上限内。
- 两个Raw评价与历史记录精确一致。
- 两组短训练与修复前已接受的entry_v2相比，初始化、episode、采样索引、优化器loss及
  policy/q权重精确一致（ordinary52步/45更新，conditional65步/58更新）。

该SUMO入口仍是工程检查，method_comparison_allowed=false、method_effect_gate=null。
它本身不覆盖200次更新；这一缺口由上述原库合成输入定期保存测试专门覆盖。
仍未运行或保证完整120k实验一定无其他问题，也不据此声称方法有效。

## 修复版正式请求（仅准备，未启动）

路径：`artifacts/rcci04c/matched_savefix_v1/request.json`。
确认hash：`54b81be7a8ebc5e80df3d139949a5fd621d1b2c04aa48a11661c75d441f87f3d`。
prepare源码：`0230f8510634da2d43b6cbb2c19fae2d482fa5aa`。

六个新模型×120k请求步=720k，四组960个评价回合，仍为20个开发场景。
旧Raw权重与预测器继续复用；没有使用失败半成品进行warm-start。
新目录仅request/preparation，无train/evaluate目录。完整运行命令见
`docs/runbooks/rcci04c_matched_policy_v1.md`；用户自行执行正式训练和评价。

原失败请求不可在新源码下继续运行。该拒绝是保留来源审计的预期行为，
不是通过改旧hash/删除旧目录规避错误。
