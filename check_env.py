# TODO: 环境检查脚本，打印依赖版本与 CUDA/GPU 信息。

import accelerate
import datasets
import matplotlib
import numpy as np
import pandas as pd
import peft
import scipy
import torch
import transformers
import trl
import yaml
import rouge_score

from src.utils import Timer, gpu_mem_mb, get_logger, set_seed


def main() -> None:
    libs = {
        "accelerate": accelerate.__version__,
        "datasets": datasets.__version__,
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "peft": peft.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "pyyaml": yaml.__version__,
        "rouge_score": getattr(rouge_score, "__version__", "unknown"),
    }
    for name, version in libs.items():
        print(f"{name}: {version}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"总显存(GB): {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f}")
    else:
        print("GPU名称: N/A")
        print("总显存(GB): 0")


if __name__ == "__main__":
    main()
