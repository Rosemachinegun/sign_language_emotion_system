from dataclasses import asdict
import importlib
import os
from pathlib import Path
import urllib.request

import cv2
import numpy as np

from config import PipelineConfig
from modules.preprocess import Preprocessor
from modules.feature_fusion import FeatureFusion
from modules.semantic_decoder import SemanticDecoder, SignPhraseRecognizer
from modules.emotion_compensation import EmotionCompensator
from modules.joint_generator import JointGenerator


class SignEmotionPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.preprocessor = Preprocessor(self.config)
        self.fusion = FeatureFusion()
        self.semantic = SemanticDecoder()
        self.emotion = EmotionCompensator()
        self.generator = JointGenerator()
        self.sign_recognizer = SignPhraseRecognizer()
        self.mp = None
        self.hands = None
        self.prev_hand_center: np.ndarray | None = None

    def _ensure_hand_model(self, force_download: bool = False) -> Path:
        model_path = Path(__file__).resolve().parent.parent / "data" / "hand_landmarker.task"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        min_valid_size = 1_000_000
        if model_path.exists() and (not force_download) and model_path.stat().st_size >= min_valid_size:
            print(f"[INFO] 手部模型已就绪: {model_path}", flush=True)
            return model_path

        if model_path.exists() and (force_download or model_path.stat().st_size < min_valid_size):
            model_path.unlink(missing_ok=True)

        default_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        env_single = os.getenv("HAND_MODEL_URL", "").strip()
        env_multi = os.getenv("HAND_MODEL_URLS", "").strip()

        urls: list[str] = []
        if env_single:
            urls.append(env_single)
        if env_multi:
            urls.extend([u.strip() for u in env_multi.split(",") if u.strip()])
        urls.append(default_url)

        print("[INFO] 正在下载手部模型（首次运行会稍慢）...", flush=True)
        last_exc: Exception | None = None
        for i, url in enumerate(urls, start=1):
            try:
                print(f"[INFO] 尝试下载源 {i}/{len(urls)}: {url}", flush=True)
                urllib.request.urlretrieve(url, model_path)
                print("[INFO] 手部模型下载完成", flush=True)
                break
            except Exception as exc:
                last_exc = exc
                print(f"[WARN] 下载失败: {exc}", flush=True)
                model_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(
                "无法下载 MediaPipe 手部模型。可设置 HAND_MODEL_URL 为镜像地址，或手动下载 hand_landmarker.task 到项目 data/ 目录。"
            ) from last_exc

        if (not model_path.exists()) or model_path.stat().st_size < min_valid_size:
            raise RuntimeError("下载的 hand_landmarker.task 文件异常（体积过小），请检查网络后重试。")
        return model_path

    def _init_hands(self) -> None:
        if self.hands is not None:
            return
        print("[INFO] 正在初始化 MediaPipe...", flush=True)
        try:
            mp = importlib.import_module("mediapipe")
            mp_tasks_python = importlib.import_module("mediapipe.tasks.python")
            mp_vision = importlib.import_module("mediapipe.tasks.python.vision")
        except ImportError as exc:
            raise ImportError("未安装 mediapipe，请先执行: pip install -r requirements.txt") from exc

        self.mp = mp
        model_path = self._ensure_hand_model()
        base_options = mp_tasks_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        try:
            self.hands = mp_vision.HandLandmarker.create_from_options(options)
        except RuntimeError as exc:
            if "Unable to open zip archive" not in str(exc):
                raise
            model_path = self._ensure_hand_model(force_download=True)
            base_options = mp_tasks_python.BaseOptions(model_asset_path=str(model_path))
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            self.hands = mp_vision.HandLandmarker.create_from_options(options)
        print("[INFO] MediaPipe 初始化完成", flush=True)

    def _extract_hand_features_and_draw(self, frame: np.ndarray) -> np.ndarray:
        if self.hands is None or self.mp is None:
            self._init_hands()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.hands.detect(mp_image)

        num_hands = 0
        openness: list[float] = []
        spread: list[float] = []
        extended_ratio: list[float] = []
        wrist_height: list[float] = []
        hand_centers: list[np.ndarray] = []
        ily_pattern_scores: list[float] = []
        open_palm_scores: list[float] = []

        hand_landmarks_list = getattr(result, "hand_landmarks", None) or []
        h, w = frame.shape[:2]

        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                num_hands += 1

                pts = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks], dtype=np.float32)
                hand_centers.append(pts[:, :2].mean(axis=0))

                pixel_pts = []
                for x_n, y_n, _ in pts:
                    px = int(np.clip(x_n * w, 0, w - 1))
                    py = int(np.clip(y_n * h, 0, h - 1))
                    pixel_pts.append((px, py))
                    cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)

                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20),
                    (0, 17),
                ]
                for s, e in connections:
                    cv2.line(frame, pixel_pts[s], pixel_pts[e], (255, 0, 255), 2)

                openness.append(float(np.linalg.norm(pts[4, :2] - pts[8, :2])))
                spread.append(float(np.std(pts[:, 0]) + np.std(pts[:, 1])))

                # 手指伸展比例（除拇指外用 tip 与 pip 的 y 比较）
                finger_up = 0
                finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
                for tip, pip in finger_pairs:
                    if pts[tip, 1] < pts[pip, 1]:
                        finger_up += 1
                # 拇指用 x 方向展开近似
                if abs(pts[4, 0] - pts[3, 0]) > 0.015:
                    finger_up += 1
                extended_ratio.append(finger_up / 5.0)

                # --- 显式手型规则：I LOVE YOU 与 open palm ---
                idx_up = 1.0 if pts[8, 1] < pts[6, 1] else 0.0
                mid_down = 1.0 if pts[12, 1] >= pts[10, 1] else 0.0
                ring_down = 1.0 if pts[16, 1] >= pts[14, 1] else 0.0
                pinky_up = 1.0 if pts[20, 1] < pts[18, 1] else 0.0

                hand_size = float(np.linalg.norm(pts[0, :2] - pts[9, :2]) + 1e-6)
                thumb_span = float(np.linalg.norm(pts[4, :2] - pts[5, :2]) / hand_size)
                thumb_open = 1.0 if thumb_span > 0.7 else 0.0

                ily_score = (idx_up + mid_down + ring_down + pinky_up + thumb_open) / 5.0
                ily_pattern_scores.append(ily_score)

                open_palm = 1.0 if (idx_up and (1.0 - mid_down) and (1.0 - ring_down) and pinky_up and thumb_open) else 0.0
                open_palm_scores.append(open_palm)

                # 手腕高度（越靠上值越大）
                wrist_height.append(float(1.0 - pts[0, 1]))

        motion = 0.0
        if hand_centers:
            center = np.mean(np.vstack(hand_centers), axis=0)
            if self.prev_hand_center is not None:
                motion = float(np.linalg.norm(center - self.prev_hand_center))
            self.prev_hand_center = center
        else:
            self.prev_hand_center = None

        return np.array([
            min(num_hands, 2) / 2.0,
            float(np.mean(openness)) if openness else 0.0,
            float(np.mean(spread)) if spread else 0.0,
            float(np.mean(extended_ratio)) if extended_ratio else 0.0,
            float(np.mean(wrist_height)) if wrist_height else 0.0,
            motion,
            float(np.max(ily_pattern_scores)) if ily_pattern_scores else 0.0,
            float(np.max(open_palm_scores)) if open_palm_scores else 0.0,
        ], dtype=np.float32)

    def run(self, video_path: str) -> dict:
        pre = self.preprocessor.process(video_path)
        fused = self.fusion.fuse(pre["frame_features"], pre["keypoint_features"])
        semantic_text = self.semantic.decode(fused)
        emotion = self.emotion.infer(pre["raw_frames"], pre["keypoint_features"])
        final_text = self.generator.generate(semantic_text, emotion)

        return {
            "config": asdict(self.config),
            "video": video_path,
            "semantic_text": semantic_text,
            "emotion": emotion,
            "final_text": final_text,
        }

    def run_webcam(self, camera_index: int = 0, mirror: bool = True) -> dict:
        print(f"[INFO] 启动摄像头模式，camera_index={camera_index}", flush=True)
        self._init_hands()

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {camera_index}")
        print("[INFO] 摄像头打开成功，窗口即将显示。按 q 退出。", flush=True)

        raw_frames: list[np.ndarray] = []
        frame_features: list[np.ndarray] = []
        keypoint_features: list[np.ndarray] = []
        phrase_features: list[np.ndarray] = []
        frame_idx = 0
        sign_guides = [
            "Try signs (easy):",
            "1) thumbs up: hold still",
            "2) goodbye: open palm wave",
            "3) I love you: ILY handshape",
        ]
        sign_display_map = {
            "点赞": "thumbs up",
            "我爱你": "I love you",
            "再见": "goodbye",
            "未识别": "unsure",
        }
        last_result = {
            "config": asdict(self.config),
            "video": f"webcam:{camera_index}",
            "semantic_text": "等待采样...",
            "emotion": {"label": "calm", "confidence": 0.0, "intensity": 0.0},
            "sign_phrase": {"label": "未识别", "confidence": 0.0},
            "final_text": "等待采样...",
        }

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if mirror:
                    frame = cv2.flip(frame, 1)

                show_frame = frame.copy()
                emotion_frame = self.emotion.infer_from_frame(show_frame)
                show_frame = self.emotion.draw_face_annotation(show_frame, emotion_frame)

                hand_feat = self._extract_hand_features_and_draw(show_frame)

                if frame_idx % self.config.frame_sample_rate == 0:
                    resized, visual_feat, _ = self.preprocessor.extract_features_from_frame(frame)
                    kp_feat = hand_feat
                    raw_frames.append(resized)
                    frame_features.append(visual_feat)
                    keypoint_features.append(kp_feat)
                    phrase_features.append(kp_feat)

                    if len(raw_frames) > self.config.max_frames:
                        raw_frames.pop(0)
                        frame_features.pop(0)
                        keypoint_features.pop(0)
                        phrase_features.pop(0)

                    if len(raw_frames) >= 8:
                        fused = self.fusion.fuse(np.vstack(frame_features), np.vstack(keypoint_features))
                        semantic_text = self.semantic.decode(fused)
                        emotion = emotion_frame
                        sign_phrase = self.sign_recognizer.predict(np.vstack(phrase_features))
                        final_text = self.generator.generate(semantic_text, emotion)
                        last_result = {
                            "config": asdict(self.config),
                            "video": f"webcam:{camera_index}",
                            "semantic_text": semantic_text,
                            "emotion": emotion,
                            "sign_phrase": sign_phrase,
                            "final_text": final_text,
                        }

                sign_cn = str(last_result["sign_phrase"]["label"])
                sign_disp = sign_display_map.get(sign_cn, "UNSURE")

                cv2.putText(show_frame, f"Emotion: {last_result['emotion']['label']}", (20, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(show_frame, f"Sign: {sign_disp}", (20, 155),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 0), 2)
                cv2.putText(show_frame, "Model: rule-based phrase recognizer", (20, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(show_frame, f"SignConf: {last_result['sign_phrase']['confidence']:.2f}", (20, 215),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)
                cv2.putText(show_frame, "Press q to quit", (20, 245),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                for i, txt in enumerate(sign_guides):
                    cv2.putText(
                        show_frame,
                        txt,
                        (20, 280 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (200, 255, 200),
                        2,
                    )

                cv2.imshow("Sign Emotion Realtime", show_frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

                frame_idx += 1
        finally:
            if self.hands is not None:
                self.hands.close()
                self.hands = None
                self.mp = None
            cap.release()
            cv2.destroyAllWindows()

        return last_result
