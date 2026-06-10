import argparse
import json
from pathlib import Path

from pipeline import SignEmotionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手语理解与情感补偿系统原型")
    parser.add_argument("--mode", choices=["0", "1"], help="运行模式：0=摄像头，1=视频")
    parser.add_argument("--video", help="输入视频路径")
    parser.add_argument("--webcam", action="store_true", help="使用电脑摄像头实时识别")
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头索引，默认 0")
    parser.add_argument("--no-mirror", action="store_true", help="关闭摄像头画面镜像")
    return parser.parse_args()


def ask_mode() -> str:
    print("\n请选择运行模式：")
    print("[0] 摄像头实时模式")
    print("[1] 加载视频文件模式")
    while True:
        mode = input("请输入 0 或 1: ").strip()
        if mode in {"0", "1"}:
            return mode
        print("输入无效，请重新输入 0 或 1。")


def ask_camera_index(default_value: int = 0) -> int:
    raw = input(f"请输入摄像头索引（默认 {default_value}）: ").strip()
    if raw == "":
        return default_value
    try:
        return int(raw)
    except ValueError:
        print("摄像头索引无效，已回退到默认 0。")
        return default_value


def ask_video_path() -> str:
    while True:
        path = input("请输入视频文件路径: ").strip()
        if Path(path).is_file():
            return path
        print("路径无效或文件不存在，请重新输入。")


def main() -> None:
    args = parse_args()
    pipeline = SignEmotionPipeline()

    mode = args.mode
    if mode is None:
        if args.webcam:
            mode = "0"
        elif args.video:
            mode = "1"
        else:
            mode = ask_mode()

    if mode == "0":
        camera_index = args.camera_index
        if not args.webcam and args.mode is None:
            camera_index = ask_camera_index(args.camera_index)
        result = pipeline.run_webcam(camera_index, mirror=not args.no_mirror)
    else:
        video_path = args.video if args.video else ask_video_path()
        result = pipeline.run(video_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
