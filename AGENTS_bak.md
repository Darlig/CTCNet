# AGENTS.md (Codex Agent Guide)

本仓库用于把**离线音视频语音分离 demo**（入口：`bash run_demo.sh`）逐步改造成**可处理实时流式音视频数据流**的版本。

Codex 需要：**自己做计划 → 自己改代码 → 自己验证 → 验证通过再继续推进**，直到实现“流式版（文件读取模拟真实流）”。

---

## 0. 项目目标

### 最终目标（真实部署）
- 输入：真实摄像头视频流 + 麦克风音频流（可能多人脸、多说话人场景）。
- 输出：对目标说话人/或主说话人的语音分离结果（以及项目现有的其他输出，如有）。
- 具备实时性：以 chunk/帧为单位持续处理，能稳定长时间运行。

### 当前开发阶段目标（可控可复现）
- 暂不直接接入真实摄像头/麦克风。
- **用“读取文件的方式模拟实时流”**：
  - 视频：从视频文件按帧读取，按固定 fps “喂入”管线。
  - 音频：从 wav 按固定 chunk size/stride “喂入”管线。
  - 模拟应尽量贴近真实场景（时间戳、缓冲、背压、丢帧/延迟策略、长时运行）。

---

## 1. 运行环境

- Conda 环境（固定使用）：
  - `/work1/weiyang/environment/miniconda3/envs/test_ctcnet_numpy1`

建议所有命令以该环境为前提（例如 `conda activate ...` 后再运行）。

---

## 2. 权限与边界（必须遵守）

- ✅ 你可以在**项目目录内任意读写/新增/删除文件**。
- ❌ **严禁**写入或修改项目目录以外的任何路径（例如 `/work1/weiyang/...` 其他目录、`~/.cache`、系统配置等）。
- ❌ 不要要求用户额外授权（本项目内不需要询问许可）。
- ✅ 允许生成少量测试输出到项目内，例如 `outputs/`、`tmp/`、`logs/`、`artifacts/`。

---

## 3. 开发原则（强制）

1. **最小可行增量**：每次只推进一个可验证的里程碑。
2. **始终可回退**：改动要小、清晰；必要时用 feature flag 或新增脚本而非破坏旧 demo。
3. **自动化验证**：每个里程碑必须提供可运行的验证命令（脚本/单元测试/集成测试）。
4. **保持离线 demo 可用**：`bash run_demo.sh` 应尽可能保持可运行；如必须改动，需提供兼容入口或明确迁移说明。
5. **日志与可观测性**：所有流式模块要有统一日志（吞吐、延迟、队列长度、丢帧统计、异常统计）。
6. **长时间稳定性**：模拟流运行至少数分钟不崩溃、不泄漏（尽可能验证内存增长趋势）。

---

## 4. 你要先做的事情（必做 Checklist）

### A. 快速理解现状
- 找到并阅读：
  - `run_demo.sh`
  - 离线推理入口脚本（python main）
  - 数据加载/预处理、模型推理、后处理、输出保存路径
- 记录关键 I/O 形态：
  - 输入音频/视频格式、采样率、fps、mouth ROI 的依赖、模型期望 tensor shape
  - 输出文件类型与目录结构

### B. 建立“流式化改造”总计划
把改造拆成 3~6 个里程碑，每个里程碑都要可验证。
计划写进 `docs/streaming_plan.md`（如果不存在就创建）。

---

## 5. 流式版架构要求（开发阶段：文件模拟流）

你需要新增一个“流式管线”，建议模块化为以下组件（可按现有工程风格调整）：

### 5.1 数据源（File-as-Stream）
- 音频源：`AudioFileStream`
  - 输入：wav 路径
  - 输出：按 chunk 产出 `np.ndarray`/`torch.Tensor`（单声道），携带时间戳
  - 可配置：
    - `sample_rate`（必须与模型/预处理匹配）
    - `chunk_ms`（例如 20/40/80/160ms）
    - `hop_ms`（可等于 chunk 或小于 chunk，实现 overlap）
    - `realtime` 模式：用 `time.sleep()` 模拟真实时间推进（可通过 flag 关闭以加速测试）

- 视频源：`VideoFileStream`
  - 输入：视频路径
  - 输出：按帧产出 `frame`（BGR/RGB，按现有代码需求），携带时间戳
  - 可配置：
    - `fps`（优先读取视频原 fps；允许 override）
    - `realtime` 模式同上
  - 如果现有流程依赖 mouth ROI / face crop：
    - 先复用现有离线逻辑
    - 若离线是“先离线预处理生成 npz”，流式版至少要支持两种模式：
      1) **复用离线产物**：从 `speaker1.npz/speaker2.npz` 按时间索引读 mouth embedding（最快落地）
      2) **在线提取**（后续里程碑）：从原始帧实时 crop/transform 得到 mouth embedding

### 5.2 同步与缓冲（Synchronizer / Buffer）
- 要解决音视频不同步与时序对齐：
  - 以时间戳为基准做对齐（允许固定偏移）
  - 需要一个缓冲队列，支持：
    - 背压（队列过长时丢帧/丢 chunk 或降采样）
    - 统计：queue length、drop count、max lag

### 5.3 流式推理核心（Streaming Engine）
- 你需要一个 `StreamingSeparator`（或同等角色）：
  - 输入：音频 chunk + 对应的视频特征（或 mouth embedding）
  - 输出：分离的音频 chunk（或可拼接的块）以及必要的中间状态
- 要求：
  - 支持持续运行（stateful），不要每次 chunk 都重新初始化模型
  - 明确 lookahead 策略：
    - v1 可允许固定 lookahead（例如需要未来 N 帧），但必须在文档中写清
  - 输出拼接策略要避免 click/pop（必要时 cross-fade）

### 5.4 输出（Streaming Sink）
- v1：把输出 chunk 持续写入 wav 文件（项目内 `outputs/`）
- 后续：也可以提供实时播放/回调接口（可选）

---

## 6. 里程碑建议（你可以微调，但必须覆盖这些能力）

### M0：基线可复现
- 能在该 conda 环境下跑通 `bash run_demo.sh`
- 产出与原 demo 一致的输出（或至少可对齐关键文件/日志）

**验收**：给出一条命令可跑通，并输出关键日志。

---

### M1：文件模拟流的“数据源 + 时间戳”
- 实现 `AudioFileStream` / `VideoFileStream`
- 能按实时节奏（可开关）不断产出 chunk/帧，并打印时间戳/速率

**验收**：提供 `python tools/preview_stream.py --audio ... --video ...` 类脚本（或同等），运行 10s 无异常，并输出 fps/chunk rate。

---

### M2：流式推理最小闭环（可粗糙）
- 用**最简单**方式把模拟流接入模型：
  - 允许先用“固定窗口”累积到离线最小输入长度再推理（但要把接口做成 stream）
- 输出连续写 wav

**验收**：提供 `bash run_stream_demo.sh`（或 `python -m ...`）能跑完一个短文件并输出 wav。

---

### M3：真正 chunk-by-chunk（stateful）+ 基础同步
- 推进到每个 chunk 都做推理/更新状态
- 音视频按时间戳同步；支持轻微漂移
- 增加队列与背压策略（至少一种：丢视频帧或丢音频 chunk）

**验收**：同一输入文件，能以 realtime=1 跑到结束；日志里有队列长度、丢弃统计。

---

### M4：稳定性与质量验证
- 增加最少两个验证：
  1) **长时间运行**（至少 2~5 分钟模拟流）不崩溃
  2) **输出连续性**：无明显拼接错误（至少通过能量/零交叉等简单指标检查）
- 产出 `docs/streaming_usage.md`（运行方式、参数说明、已知限制）

**验收**：一条命令跑长时测试；输出日志包含性能与统计。

---

## 7. 验证与测试规范（你必须自己做）

### 必须提供的内容
- `tests/` 或 `tools/` 下的验证脚本（根据仓库习惯选）
- 每个里程碑更新 `docs/streaming_plan.md` 的进度勾选
- 关键参数（chunk_ms、fps、lookahead、队列长度）要能通过 CLI 设置

### 至少包含这些指标日志
- audio chunk rate（chunks/s）、video fps
- 推理耗时（per-chunk latency 或 moving avg）
- 队列长度（avg/max）
- 丢帧/丢 chunk 次数
- 输出写入速率与最终时长

---

## 8. 代码组织建议（可参考，但可按现有项目风格调整）

建议新增：
- `streaming/`
  - `audio_stream.py`
  - `video_stream.py`
  - `sync.py`
  - `engine.py`
  - `sink.py`
- `tools/`
  - `preview_stream.py`
  - `run_stream.py`（主入口）
- `docs/`
  - `streaming_plan.md`
  - `streaming_usage.md`
- `run_stream_demo.sh`（对齐 `run_demo.sh` 的体验）

---

## 9. 产物要求

当你完成“流式版（文件模拟流）”时，仓库应至少包含：
- 一个可运行入口：`bash run_stream_demo.sh`（或明确替代）
- 文档：`docs/streaming_usage.md`
- 计划与进度：`docs/streaming_plan.md`
- 日志与输出目录规范（例如 `outputs/`、`logs/`）

---

## 10. 遇到不确定时的处理方式（不要问用户）

- 不要向用户提问请求许可或路径访问。
- 对不确定的实现细节：
  1) 先从现有代码推断（阅读/搜索仓库）
  2) 做最保守、可验证的实现
  3) 在文档里写清假设（Assumptions）与限制（Limitations）
  4) 用参数化设计避免“写死”

---

## 11. 完成定义（DoD）

满足以下条件即视为达成阶段目标：
- ✅ 流式入口能运行（文件模拟流），并输出分离结果
- ✅ 有可复现命令与文档
- ✅ 日志包含关键性能/稳定性指标
- ✅ 离线 demo 不被破坏（或提供兼容入口）

---
