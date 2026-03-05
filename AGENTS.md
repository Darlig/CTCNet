# AGENTS.md (Codex Agent Guide) — v2 End-to-End Streaming AV Pipeline

本仓库目标：把现有**离线音视频语音分离 demo**（入口：`bash run_demo.sh`）改造成**可处理实时流式音视频数据流**的版本。

当前进展：你已实现 **Streaming Separator**（输入：混叠音频流 + 目标人物 mouthroi/特征流 → 输出：目标人分离音频流）。  
下一阶段：把 demo 中 **离线 mouthroi 生成脚本**（入口：`bash run_prepare_audio_mouth.sh`）整合进来，实现 **从原始视频+原始音频 → 在线/流式生成 mouthroi → 流式分离输出** 的端到端流程。  
开发阶段仍用“读文件模拟实时音视频流”，但模拟必须尽量贴近真实摄像头/麦克风。

---

## 0. 最终目标与当前阶段目标

### 最终目标（真实部署）
- 输入：真实摄像头视频流 + 麦克风音频流（可能多人脸/多人声）。
- 输出：目标说话人的分离音频（以及项目已有的其它输出如适用）。
- 长时间稳定运行；具备可观测性（延迟/吞吐/队列/丢帧统计）。

### 当前开发阶段目标（文件模拟实时流，必须贴近真实）
- 用文件模拟“实时到达”的音视频流：
  - 视频：按帧读取并按 fps 推进时间（可 `--realtime` 开关）。
  - 音频：按 chunk/hop 读取并按时间推进（可 `--realtime` 开关）。
- **禁止**把整个视频/音频一次性离线预处理完再喂给流式分离器（除非作为调试模式，并且默认关闭）。

---

## 1. 运行环境

- 固定 Conda 环境：
  - `/work1/weiyang/environment/miniconda3/envs/test_ctcnet_numpy1`

---

## 2. 权限与边界（必须遵守）

- ✅ 可以在**项目目录内**任意读写/新增/删除文件。
- ❌ 严禁修改项目目录外任何路径（含系统配置、用户目录、其它 work1 子目录等）。
- ✅ 允许写入项目内：`outputs/`, `logs/`, `tmp/`, `artifacts/`（如不存在可创建）。

---

## 3. 强制开发原则（必须执行）

1. **自规划 → 自实现 → 自验证 → 验证通过再推进**。
2. **最小可行增量**：每次推进一个里程碑，改动小、可回退。
3. **保持离线 demo 可用**：`bash run_demo.sh` 尽量不破坏；若必须变更，提供兼容入口/说明。
4. **Streaming 约束**：
   - 所有模块必须支持**增量处理**，不得依赖“完整序列上下文”。
   - mouthroi 生成必须 streaming-compatible：按帧处理、维护状态、输出逐帧（或逐窗口）特征。
5. **可观测性**：所有关键阶段输出统一日志（吞吐、延迟、队列长度、drop 计数）。
6. **稳定性**：模拟流可运行 ≥ 5 分钟（至少在最终里程碑达成时）。

---

## 4. 端到端流式 Pipeline 规范（v2）

你要实现/维护的最终（开发阶段）流水线如下：

```
Raw Video File + Raw Audio File
        │
        ├── VideoFileStream (frames with timestamps)
        │         │
        │         ▼
        │   MouthROIExtractor (streaming)
        │         │
        │         ▼
        │   Mouth Feature/Embedding Stream (timestamped)
        │
        └── AudioFileStream (audio chunks with timestamps)
                  │
                  ▼
        AVSynchronizer / Buffer (align A/V by timestamps, backpressure)
                  │
                  ▼
        StreamingSeparator (stateful, chunk-by-chunk)
                  │
                  ▼
        StreamingSink (write continuous wav, optional metrics)
```

### 必须支持的两种 mouthroi source（用于调试与对照）
- Mode A（调试/对照）：读取已有的离线 mouthroi 产物（npz 等），按时间索引“流式吐出”。
- Mode B（默认/目标）：从原始视频帧**在线生成** mouthroi/特征（流式）。

两种 mode 通过 CLI/配置切换，默认使用 Mode B。

---

## 5. 组件要求（你可按仓库风格组织，但能力必须覆盖）

### 5.1 AudioFileStream（文件模拟音频流）
- 输入：wav
- 输出：带 timestamp 的 audio chunk（支持 overlap）
- 参数：`--sample-rate`, `--chunk-ms`, `--hop-ms`, `--realtime`

### 5.2 VideoFileStream（文件模拟视频流）
- 输入：video
- 输出：带 timestamp 的 frame
- 参数：`--fps`（优先读取源 fps，可 override）, `--realtime`

### 5.3 MouthROIExtractor（流式）
- 复用你将提供的离线脚本逻辑，但必须改造成 streaming：
  - 输入：逐帧 frame + timestamp
  - 输出：逐帧 mouthroi（或 mouth embedding）+ timestamp
- 允许维护跨帧状态（如追踪、缓存），但不得要求一次性读完整视频。

### 5.4 AVSynchronizer / Buffer（对齐与背压）
- 以 timestamp 对齐音频 chunk 与视频特征
- 提供背压策略（至少一种）：
  - 队列过长时：丢视频帧 / 丢音频 chunk / 降采样（必须记录 drop 计数）
- 输出：
  - 对齐后的 (audio_chunk, mouth_feat) 给 separator

### 5.5 StreamingSeparator（已实现，需保持接口清晰）
- stateful：不要每个 chunk 重新初始化模型/状态
- 明确 lookahead/latency 策略并写入文档（可 v1 允许固定 lookahead）

### 5.6 StreamingSink
- v1：持续写 wav 到 `outputs/`
- 处理 chunk 拼接：避免明显 click/pop（必要时 cross-fade）

---

## 6. 里程碑计划（v2）

你必须以里程碑推进，并在 `docs/streaming_plan.md` 勾选进度（没有就创建）。

### M0：离线基线可复现
- `bash run_demo.sh` 可跑通并产出输出
- 记录关键 I/O shape、采样率、fps、依赖文件

**验收**：一条命令 + 关键日志。

### M1：文件模拟流数据源可用
- AudioFileStream / VideoFileStream 输出稳定（10s 无异常）
- 打印 chunk rate / fps / timestamp

**验收**：`python tools/preview_stream.py ...`（或同等）可跑。

### M2：现有 Streaming Separator 闭环回归
- 使用 mouthroi Mode A（读离线 mouthroi 产物）跑通：
  - 视频/音频文件 → 同步对齐 → separator → 输出 wav

**验收**：`bash run_stream_demo.sh --mouthroi-mode precomputed ...` 可跑完短样本。

### M3：MouthROIExtractor 流式化（核心新增）
- 把离线 mouthroi 生成脚本改造成 streaming component：
  - frame-in → mouthroi/feat-out（逐帧/逐窗口）
- 与时间戳对齐逻辑清晰（必要时引入固定 offset 参数）

**验收**：`python tools/preview_mouthroi_stream.py ...`（或同等）可跑并输出统计：
- 输出帧率、丢帧数、平均耗时、生成的 feat shape。

### M4：端到端流式（默认走在线 mouthroi）
- 默认 Mode B：
  - Raw video + raw audio → 在线 mouthroi → 对齐 → separator → 输出 wav
- 严禁默认离线预处理整段视频（可以提供 debug flag，但默认关闭）

**验收**：`bash run_stream_demo.sh ...` 默认配置可跑完短样本。

### M5：稳定性与性能指标
- 运行 ≥ 5 分钟模拟流（可用循环播放文件/拼接长文件）
- 输出日志包含：
  - 推理耗时（moving avg/p95）
  - A/V 队列长度（avg/max）
  - drop 计数
  - 输出音频时长一致性检查（输入时长 vs 输出时长）

**验收**：`bash tools/run_long_stream_test.sh`（或同等）可一键跑。

### M6：文档与可用性收口
- `docs/streaming_usage.md`：
  - 如何运行端到端流式 demo
  - 参数解释（chunk/fps/lookahead/offset/backpressure）
  - 已知限制与下一步（真实摄像头/麦克风接入）

**验收**：文档清晰 + 命令可复现。

---

## 7. 验证与测试规范（必须提供）

### 每个里程碑至少提供一个可执行验证入口
- `tools/` 下脚本或 `tests/` 下集成测试均可。
- 重要：验证命令必须在上述 conda 环境中可跑。

### 必须记录的日志指标（每次流式运行都应打印/保存）
- audio chunk rate（chunks/s）
- video fps
- mouthroi 生成耗时（avg / max）
- separator 推理耗时（avg / p95）
- 队列长度（avg / max）
- drop counts（audio/video/feat）
- 输出 wav 写入统计（累计时长、文件路径）

---

## 8. 推荐目录结构（可按仓库风格调整）

建议新增（如不存在）：
- `streaming/`
  - `audio_stream.py`
  - `video_stream.py`
  - `mouthroi_stream.py`
  - `sync.py`
  - `engine.py`
  - `sink.py`
- `tools/`
  - `preview_stream.py`
  - `preview_mouthroi_stream.py`
  - `run_stream.py`（主入口）
  - `run_long_stream_test.sh`
- `docs/`
  - `streaming_plan.md`
  - `streaming_usage.md`
- `run_stream_demo.sh`

---

## 9. 不确定时的处理方式（不要问用户）

- 不要请求许可或询问是否能写某目录（项目内都可以写）。
- 对不确定实现细节：
  1) 优先读仓库现有代码推断接口
  2) 采取最保守可验证实现
  3) 在 `docs/streaming_usage.md` 记录假设与限制
  4) 通过 CLI 参数化避免写死

---

## 10. 完成定义（DoD）

达成以下即认为“端到端流式（文件模拟）”完成：
- ✅ 默认 Mode B（在线 mouthroi）端到端可跑：video+audio → separated wav
- ✅ 具备关键日志与 ≥5 分钟稳定性验证
- ✅ 文档齐全（usage + plan）
- ✅ 离线 demo 不被破坏（或有兼容入口）
