# RCCI01 既有数据与道路几何可计算性审计验收

日期：2026-10-17（Asia/Shanghai）。
工作树：E:/Prediction_RL_RCCI。
分支：codex/road-conflict-consequence-interface-v1。
文档父提交：dca9136453611e1ea4f324fdbeb503f1da85b8ef。
实现提交：c31f09d；兼容修复/实际运行提交：228ac2ca05f9813b748a839c467c534468591180。

## 交付边界

完成 RCCI01 的数据可用性、来源和已知道路几何审计入口及实际运行。
不是线上道路后果 adapter、forecast 投影/事件误差验收或方法效果实验。
不修改原研究方案、Raw provider、预测器/损失、环境、奖励、DDPG、动作、
已有配置、seed 或结果。源码从本工作树读取，基线仅作文件输入。
没有导入归档差分/课程线，没有训练、预测器前向或 SUMO 仿真回合。

新增人工配置独立可读，无 extends。新脚本只允许 train/validation，
保留原 review 顺序及空监督根；不按输赢或事件删根。
来源是 713/177 个已有可读取根，不冒称所有预期但未达到的 target 均有数据。
calibration/test 轨迹不打开；读取的冻结 review 允许包含其 split 元数据。
模型只核对两份固定 ensemble SHA，不假称已完成权重加载/runtime 复用验收。

## 实际运行的测试

- 新增测试最终为 24 passed。
- 相关离线回归最终为 195 passed，包含上述 24 项；4 项单独 deselected。
- 首次包含原生契约导入的回归实际为 194 passed / 4 failed，原因均为新
  worktree 未包含被 Git 忽略的 st_cy 编译扩展；没有修改原测试、关闭
  USE_CYTHON、安装包或向基线写入扩展来掩盖失败。原生构建留 RCCI03。
- 最终命令为以下九个测试文件，并使用
  -k 'not test_actual_upstream_reward_done_and_cleanup'：
  test_rcci_feasibility.py、test_probe_features.py、test_dataset_contract.py、
  test_candidate_ranking.py、test_task_relevance.py、test_response_collection.py、
  test_masked_training.py、test_all_ddpg_features.py、test_training_contract.py。
- 首次新测试临时目录缺少父目录导致一次 setup error；创建本工作树下
  artifacts/tests 后解决。没有修改测试语义或清除已有实验目录。
- git diff --check 和 staged diff 检查通过，无模型/数据/日志进入提交。

## 首次预检查的兼容问题

rcci01_bnd_v1 在 prepare 创建产物前被地图字节哈希检查拒绝，未运行审计。
主项目 XML SHA 为 47874cfb6faa044253f065bb9bf1ab56b24c9ba6aeccd24b1449af21dfffe9f1，
工作树 XML SHA 为 8c7b40f28bb6fc2bedf8e3b958bff88d1ff8cbe43afa13cce9d0550020193b60。
Git 显示地图没有内容修改，差异仅为 LF/CRLF。修复后分别将两份检出绑定
到 75051acc 的同一 Git blob，仅允许换行差异，拒绝 XML 语义/空白修改。
新请求还绑定本工作树实际地图字节 SHA。没有更改地图、旧报告或历史 guard。

## 已完成的有界审计

```powershell
python -B tools/audit_rcci_feasibility.py audit --baseline-root E:/Prediction_RL --run-id rcci01_bnd_v2
```

train/validation 各前三根、共 30 个已存分支；每个 split 仅一个源 episode，
不据此推断整体覆盖。状态 complete。
请求 hash：0b1a7b82f7b1109b6325c094d26445163cfe35f4f69c1d77fb2e233254e335e5。
report SHA：fe89db8682f4fab60f0ce48fa6ab47017c244caccafc1a7ec983d6cc13df6a73。

## 已完成的完整 train/validation 审计

路径：artifacts/rcci01/rcci01_full_v1。
请求 hash：25e5b0d195c6a41505969e5fb20f25dfd3b1188600994b15c1565eeecb20bc2d。
report SHA：47665549cb455cd214a878aa8cae8985cbcb82c5aebf47bc401c59421d623fe3。
roots SHA：777836b6a497fbbfaa028a8608dffa556a4014108e5a61da274e6b7dc8b685bb。

| 描述量 | train | validation |
| --- | ---: | ---: |
| 已有根数 | 713 | 177 |
| 源 episode 数 | 256 | 64 |
| 源 simulator seeds | 11000–11255 | 12000–12063 |
| 已存候选分支 | 3565 | 885 |
| root + 非终止 future actor 状态计数 | 1338206 | 341913 |
| 时域截尾分支 | 2159 | 510 |
| 分支内出现 root 后新 actor | 3243 | 800 |
| 存在容量外车辆的根 | 713 | 177 |
| 记录了航向的状态 | 0 | 0 |
| 已知 lane 投影最大残差 / m | 5.68e-14 | 2.84e-14 |
| lanePosition 重建 XY 最大残差 / m | 2.85e-14 | 2.85e-14 |
| 观测到的 lane 转换 | 12113 | 3043 |
| 不在原四个采样下标的 lane 转换 | 10384 | 2632 |

状态计数含同一车辆在不同根/候选/时间的重复，不是独立样本量。
已知 lane 是离线原生标签；近浮点误差的坐标重建只证明坐标变换自洽，
不证明预测轨迹能被正确投影，更不是新接口预测准确性或安全性的结果。
稀疏采样下标之外的 lane 转换不是漏检冲突，单独不能解释旧策略的失败。

全部根都有容量外车辆、很多分支有新 actor，只说明全场景精确覆盖不成立，
并不证明每根都遗漏关键冲突。root summary token 没有可定位轨迹，不把它
当成车辆；新 actor 的未来存在性也不能作为在线 mask 输入。
约 60% 分支截尾，不将未观测尾部当无冲突。未记录 angle/heading，不把
后续根据道路切向/位置差构造的车身框标为原生 footprint 真值。

运行后再次只读核验全部 6556 个输入文件哈希、请求/输出 receipt、根族
顺序、分组计数、均值/最大值/直方图与逐根记录一致；source/environment
身份不变，writer.lock 正常释放，基线和本工作树均干净。全局 quantile
来自实际数值数组，不用逐根 quantile 的平均值代替。

## 环境与原生运行限制

请求记录 Python 3.10.16、PyTorch 2.5.1、NumPy 2.2.6、ALL 0.5.3、
TraCI/sumolib 1.25.0，Windows 10.0.26200。
另以只读 --version 检查发现 SUMO executable 为 1.22.0：
E:/Program Files/sumo-1.22.0/bin/sumo.exe，SHA256
f4732c44fa22fbfd327a336a07ee95dec6e628ede390e10fd44554860403988c。
这不是 SUMO 仿真或 native 兼容验收。客户端/可执行文件版本不同不能仅凭
版本号认定不兼容，也不能直接认定兼容；与 st_cy 构建一起留在 RCCI03。
没有安装、升级依赖或修改全局环境。

## 审阅结论与下一阶段

可以进入 RCCI02 的契约设计，但不能据此开始策略训练或宣称 RCCI 有效。
需要先确定：

1. 有限合流冲突区域的地图定义。两条路径进入共同下游后长期重合，不能
   把整个共同下游简单当成交叉冲突区；下游跟驰净距与合流交汇应分开。
2. 预测轨迹在 root 合法路径上的投影、歧义、离路和网络端点处理；不读取
   future lane，不强制吸附成安全道路状态。
3. 航向/车身 extent 的明确近似及适用 mask；已知 lane 几何不自动等于
   原生完整车身占用。任何近似都要与精确真值的用语区分。
4. 完整 25 点的占用窗口/次序/裕度、时域截尾和有效性编码，再冻结定长
   布局、尺度和容量匹配。不能将真实未来终止 mask 带入在线接口。

forecast 投影覆盖率、后果误差、动作条件信息来源分解以及线上接入均未
实施，仍须在接口契约确定后完成。三成员后果聚合保持可选，critic/课程/
差分监督/新奖励/Shield/新场景仍不在当前范围。没有发布训练请求。

只在 RCCI 分支提交，未合并 main、未推送、未创建 tag。主项目及两个归档
研究工作树不变；未向旧 ACCVP 项目写入任何文件。
