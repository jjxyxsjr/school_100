# src/main.py

import torch
import os

# 从其他模块导入我们定义的类
from data_loader import MNISTDataLoaders
from model import FCNN
from trainer import ModelTrainer


from utils import plot_training_history # 修正后的名称 # 如果你创建了绘图工具函数

def run_training():
    """
    主函数，用于配置和启动 MNIST FCNN 模型的训练过程。
    """
    # --- 1. 配置参数 ---
    # 路径配置 (相对于项目根目录)
    # 注意：当从 src/ 目录运行 main.py 时，这些路径可能需要调整，
    # 或者更好地，将 main.py 移到项目根目录，或者使用绝对路径/更鲁棒的路径管理。
    # 假设我们从项目根目录运行 (例如 python src/main.py)，或者IDE配置正确。
    # 为了简单起见，我们假设 data_dir 和 saved_models_dir 是相对于项目根目录的。
    # 如果你在 src 目录下运行 `python main.py`，那么 '../data' 是正确的。
    # 如果你在项目根目录下运行 `python -m src.main`，那么 './data' 是正确的。
    # 我们这里先用相对于 src 目录的路径。

    # 获取当前 main.py 脚本所在的目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录通常是 src 目录的上一级
    project_root_dir = os.path.dirname(current_script_dir)

    DATA_DIR = os.path.join(project_root_dir, 'data')
    SAVED_MODELS_DIR = os.path.join(project_root_dir, 'saved_models')
    RESULTS_DIR = os.path.join(project_root_dir, 'results')

    # 确保保存模型的目录存在
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 训练超参数
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 10  # 可以先设小一点测试，例如 5 或 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用的设备: {DEVICE}")

    # 模型参数 (与 model.py 中的 FCNN 定义匹配)
    INPUT_SIZE = 784  # 28*28
    HIDDEN_SIZE1 = 512
    HIDDEN_SIZE2 = 256
    NUM_CLASSES = 10

    # --- 2. 加载数据 ---
    print("加载 MNIST 数据集...")
    mnist_data_loaders = MNISTDataLoaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE)
    train_loader = mnist_data_loaders.get_train_loader()
    # 通常我们会有一个单独的验证集，这里为了简单，我们使用测试集作为训练过程中的验证集
    # 在实际项目中，最好从训练集中划分一部分作为验证集
    val_loader = mnist_data_loaders.get_test_loader()
    print("数据加载完成。")

    # --- 3. 初始化模型 ---
    print("初始化 FCNN 模型...")
    model = FCNN(
        input_size=INPUT_SIZE,  # 虽然 FCNN 内部有 Flatten，但明确指定可以避免混淆
        hidden_size1=HIDDEN_SIZE1,
        hidden_size2=HIDDEN_SIZE2,
        num_classes=NUM_CLASSES
    )
    # model 将在 Trainer 内部被移到 DEVICE
    print("模型初始化完成。")
    # print(model) # 可以取消注释查看模型结构

    # --- 4. 初始化训练器 ---
    # 优化器和损失函数在 ModelTrainer 中有默认值 (Adam, CrossEntropyLoss)
    # 如果需要自定义，可以在这里创建实例并传入
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # criterion = torch.nn.NLLLoss() # 如果模型输出是 log_softmax

    print("初始化模型训练器...")
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,  # 将测试加载器用作验证加载器
        device=DEVICE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE  # 如果 ModelTrainer 中的 optimizer=None，这个lr会生效
        # optimizer=optimizer, # 如果自定义了优化器
        # criterion=criterion, # 如果自定义了损失函数
    )
    print("训练器初始化完成。")

    # --- 5. 开始训练 ---
    print("开始训练模型...")
    training_history = trainer.train()  # train() 方法返回一个包含历史记录的字典
    print("模型训练完成。")

    # --- 6. 保存训练好的模型 ---
    final_model_path = os.path.join(SAVED_MODELS_DIR, "mnist_fcnn_final.pth")
    trainer.save_model(final_model_path)  # ModelTrainer 有一个 save_model 方法
    print(f"最终模型已保存到: {final_model_path}")

    # --- 7. (可选) 保存和绘制训练历史 ---
    # 你可以将 training_history (包含损失和准确率列表) 保存到文件
    # 或者使用 matplotlib 绘制曲线图并保存
    print("\n训练历史:")
    for key, values in training_history.items():
        print(f"{key}: {values}")

    # 示例：如何使用 matplotlib 绘制 (如果你有 utils.py 和绘图函数)
    from utils import plot_training_history
    plot_file_path = os.path.join(RESULTS_DIR, "training_history_curves.png")
    plot_training_history(training_history, save_path=plot_file_path)
    print(f"训练曲线图已保存到: {plot_file_path}")


if __name__ == '__main__':
    run_training()