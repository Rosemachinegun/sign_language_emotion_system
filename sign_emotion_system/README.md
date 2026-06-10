# 基于大模型的手语理解与情感补偿系统（Python原型）

该项目根据你的开题报告实现了一个可运行的**原型架构**，对应“感知-理解-补偿-输出”流程：

1. 多模态输入与预处理（视频帧、关键点占位）
2. 多模态特征提取与融合（时序统计融合）
3. 语义理解（可替换为真实大模型）
4. 情感补偿（基于表情强度的简化估计）
5. 语义-情感联合生成（情感增强文本）
6. 输出（CLI JSON）

当前实时版已支持：
- 人脸框 + 表情标注：`smile` / `calm`
- MediaPipe 手部 21 点关键点与连线可视化
- 窗口内提示可练习的短手语语句（你好、谢谢、我爱你、对不起、再见）

## 目录

- `src/main.py`：命令行入口
- `src/pipeline.py`：总流程编排
- `src/modules/preprocess.py`：视频读取与关键点占位提取
- `src/modules/feature_fusion.py`：视觉与关键点特征融合
- `src/modules/semantic_decoder.py`：语义解码（基线实现）
- `src/modules/emotion_compensation.py`：情感识别与补偿特征
- `src/modules/joint_generator.py`：联合生成

## 快速开始

```bash
cd sign_emotion_system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py --video /path/to/sign_video.mp4
```

也可直接无参数启动，按 `0/1` 选择模式：

```bash
python src/main.py
```

> 若没有视频，可先创建一个短视频测试。系统会输出结构化 JSON 结果。

## 实时摄像头模式

```bash
cd sign_emotion_system
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py --webcam
```

也支持显式模式参数：

```bash
# 0=摄像头
python src/main.py --mode 0 --camera-index 0

# 1=视频
python src/main.py --mode 1 --video /path/to/sign_video.mp4
```

> 首次运行会自动下载 MediaPipe 手部模型到 `data/hand_landmarker.task`。
> 若网络受限，请手动下载该文件放到 `data/` 目录后再运行。

若你在国内网络环境，可先配置镜像下载地址：

```bash
export HAND_MODEL_URL="https://你的镜像地址/hand_landmarker.task"
python src/main.py --webcam
```

也支持多个候选地址（逗号分隔），程序会按顺序尝试：

```bash
export HAND_MODEL_URLS="https://镜像1/hand_landmarker.task,https://镜像2/hand_landmarker.task"
python src/main.py --webcam
```

- 摄像头模式默认开启**镜像显示**（更接近常见自拍预览）。
- 若需关闭镜像：

```bash
python src/main.py --webcam --no-mirror
```

- 默认使用 `0` 号摄像头；如有多个摄像头：

```bash
python src/main.py --webcam --camera-index 1
```

- 按 `q` 退出实时窗口，终端会打印最后一次识别结果 JSON。

## 实时演示建议（易做易识别）

建议按以下短句对应动作进行演示：
- 你好
- 谢谢
- 我爱你
- 对不起
- 再见

同时配合表情切换验证：
- 微笑（应更容易显示 `smile`）
- 放松无表情（应更容易显示 `calm`）

## 后续升级建议

- 将 `semantic_decoder.py` 替换为 `LLaMA/ChatGLM + LoRA` 推理服务。
- 将 `preprocess.py` 的关键点占位提取替换为 `MediaPipe` 真实关键点。
- 将 `emotion_compensation.py` 接入真实表情识别网络（如 AffectNet 迁移模型）。
- 增加 Web API（FastAPI）和前端可视化界面。
