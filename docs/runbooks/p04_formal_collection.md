# P4g 正式预测数据采集（由用户手动启动）

当前状态：采集代码、共享核心的开发验收、正式请求准备已完成。
**正式数据尚未采集，也没有开始预测器或 DDPG 训练。**

## 本轮范围

独立配置：`configs/formal/p04_response_dataset_v1.json`，不使用 extends。
保留 `p04_response_dataset_v1_draft.json` 原文件，不覆盖历史草案。
除协议名称、手动启动状态和执行保护外，数据/模型/训练/校准参数与草案一致；
单元测试对此逐字段检查。配置中保留的训练/校准部分是后续设计，不代表相应
完整训练器、校准器或 P5 排序评价已经实现。

| 划分 | Simulator seeds | Episodes | 最大根状态数 | 最大独立候选数 |
| --- | --- | ---: | ---: | ---: |
| Train | 11000–11255 | 256 | 768 | 3840 |
| Validation | 12000–12063 | 64 | 192 | 960 |
| Calibration | 13000–13063 | 64 | 192 | 960 |
| Test（继续锁定） | 14000–14127 | 128 | 384 | 1920 |

当前可手动采集前三部分，共 384 episodes，最多 1152 个根状态、5760 个独立
候选，另有一次逆序重复，即最多 11520 次候选分支执行。重复不是独立样本。
未到达的根状态如实记录，不补 seed、不移动位置、不凑满数量。

仅采集每个 seed 的第一个 episode。参考前缀由原作者 DDPG 生成；候选施加
一个原始控制周期的 jerk，之后固定零 jerk。原始奖励、连续控制、SUMO 设置
和终止语义保持不变。只使用阻塞同步，不增加 Shield 或 wall-clock stale。

## 已准备的请求：现在无需再次 prepare

工作目录 `E:/Prediction_RL`，使用 `pytorch` 环境。

- 请求：`artifacts/data/response_v1/request.json`
- 准备记录：`artifacts/data/response_v1/preparation.json`
- 状态：`prepared_not_executed`，正式 episode 目录数为 0。
- 确认哈希（规范化完整 request 的 SHA256，不是文件字节哈希）：
  `ed1a90bfeff0c3ab01fa4841ca7a8fa1588e77799aaf1087175916e36210f9a6`

阅读配置及本页后，执行以下命令表示确认该具体请求并手动启动长时间采集：

```powershell
python tools/collect_response_dataset.py run --request artifacts/data/response_v1/request.json --splits train validation calibration --confirm-request-hash ed1a90bfeff0c3ab01fa4841ca7a8fa1588e77799aaf1087175916e36210f9a6 --resume
```

这是采集命令，不是模型训练命令。可按相同请求分批运行：第一次仅传
`--splits train`，以后传 `--splits validation calibration`。不能传 test 或 development。
不建议同时启动多个采集进程；单写锁及单 worker 为当前已验收执行方式。

每个新 episode 启动独立 Python/SUMO worker；进度显示 job、split、seed 和
完成数量。每个 worker 的 300 秒上限只用于工程异常检测，绝不将超时结果
转换成 stale observation。它不是实时部署声明，也不是保证所有新场景都能
在此时间内结束。超限或语义异常会停止并保留证据，不能盲目放宽上限。

采集过程中不要修改源码、配置或输入证据；请求绑定这些内容及环境版本。
当前正式集合尚未实际跑过，正式覆盖率和耗时未知。开发 seed 的耗时不能
当作正式采集的完成时间保证。

## 输出和复用规则

- `e0000..e0255`：train；`e0256..e0319`：validation；`e0320..e0383`：calibration。
- test job 仅存在于锁定清单；本入口无法执行，也不会创建 test episode 数据。
- 每个 episode 保存原始 settings、root discovery、根状态 accounting、历史输入、
  完整分支审计、未来几何 sidecar、标签和不可变 `complete.json` 文件清单。
- 历史只含已观察数据；未来车道/尺寸信息和终止屏蔽的标签独立保存。
- `invocations/<id>/authorization.json` 保存用户本次启动的请求哈希、split 和 jobs。
- `invocations/<id>/report.json` 是本次执行/复用/失败及分 split 汇总结果。
  终端会打印完整路径；每次调用保留自己的报告，不覆盖旧结论。

`--resume` 只复用通过完整文件哈希、身份、候选族和监督契约验证的已完成
episodes，然后执行本次所选 split 内尚未开始的 jobs。它不会降低检查标准。
父进程开始及每个 episode 完成前检查输入绑定；配置/环境/源码改变时拒绝继续。
Python 源码按 LF 规范化，避免 Git 的 CRLF 转换导致无意义失配。

失败或中断产生的**不完整 episode 不会自动重试**，其他完整 episode 也不会
被删除。若遇到错误，请先保留并提供 invocation 报告、对应 job 日志和
`failure.json`（若存在），进行诊断；不要直接删文件、换 seed 或覆盖请求。
若因父进程硬退出遗留 `writer.lock`，先确认没有关联的 Python/SUMO 在运行，
再讨论恢复，工具不会自动认定锁过期。当前版本不提供部分 episode 修复命令。

## 采集完成后先审查数据，再训练

`status=complete` 只表示本次指定 jobs 的工程流程完成，不表示预测质量有效。
`formal_training_ready=false` 和 `coverage_review_required=true` 是有意保留的。
须检查每个 split、每个位置的到达/缺失及原因、无根状态 episodes、终止事件、
空轨迹候选、有效邻车-时间监督单元、actor 遗漏与响应幅度。
全部碰撞或全部根状态不可达同样会如实汇总，不被伪装成程序失败或成功效果。

将完成报告用于数据覆盖审查，再决定能否进入完整普通/动作条件三成员预测器
训练与校准。预测器训练、P5 验收、DDPG 特征接入及正式方法比较仍是后续步骤。
测试数据须在预测器与校准冻结后通过另一项明确的测试集释放实现，不能通过
编辑当前请求解除锁定。

## 需要重新准备独立请求时

已有请求不可覆盖；不要重复准备 response_v1。仅在有明确原因且源码未改变时，
可用新的短 run ID；任何代码改动需重新验收共享核心，不能复用过时的验收结果。

```powershell
python tools/collect_response_dataset.py prepare --config configs/formal/p04_response_dataset_v1.json --collector-report artifacts/p4/p4_collect_v2_01/invocations/398fb383670c/report.json --run-id response_v1_repeat
```

prepare 不运行任何仿真/训练，也不会加载策略；它只验证证据、模型文件哈希、
环境版本并生成请求。后续手动 run 必须使用它实际打印的完整确认哈希。
