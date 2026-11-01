# game-solver

使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 管理 Python 依赖，需要 Python 3.11 或更高版本。

```sh
uv sync --locked
uv run python -m unittest discover -s arrows/tests
uv run python arrows/solve.py --device 192.168.2.174:43345 --play
```

箭头游戏的设备连接和运行选项见 [arrows/README.md](arrows/README.md)。旧版 TensorFlow 图像识别脚本如需使用，可执行 `uv sync --locked --extra legacy-ocr` 安装可选依赖。ADB 及可选的 Tesseract 是系统工具，需另行安装。
