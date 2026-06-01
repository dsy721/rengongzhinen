# TODO: 包初始化文件，确保在导入 HF 相关库前优先设置国内镜像环境变量。

from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
