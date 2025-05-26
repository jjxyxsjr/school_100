import torch
import subprocess

# PyTorch & CUDA 编译版本信息
print(f"PyTorch 版本: {torch.__version__}")
print(f"编译时使用的 CUDA 版本: {torch.version.cuda}")
print(f"cuDNN 版本: {torch.backends.cudnn.version()}")

# 系统级 CUDA 驱动版本（nvidia-smi）
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
    print("\nnvidia-smi 输出:")
    print(result.stdout)
except FileNotFoundError:
    print("未找到 nvidia-smi，可能未安装 NVIDIA 驱动或环境变量未配置。")
except subprocess.CalledProcessError as e:
    print("运行 nvidia-smi 时出错：", e)

# GPU 检测与测试
print(f"\nGPU 可用状态: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"当前设备: {torch.cuda.get_device_name(0)}")
    x = torch.rand(3, 3).cuda()
    print(f"\n随机张量:\n{x}\n张量所在设备: {x.device}")
else:
    print("未检测到可用的 GPU，张量将在 CPU 上运行。")
