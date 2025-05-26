# src/model.py

import torch
import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, input_size: int = 784, hidden_size1: int = 512, hidden_size2: int = 256, num_classes: int = 10):
        """
        一个简单的3层全连接神经网络 (FCNN) 用于 MNIST 分类。

        参数:
            input_size (int): 输入特征的数量 (对于展平的 MNIST 是 28*28=784)。
            hidden_size1 (int): 第一个隐藏层中的神经元数量。
            hidden_size2 (int): 第二个隐藏层中的神经元数量。
            num_classes (int): 输出类别的数量 (对于 MNIST 是 10)。
        """
        super(FCNN, self).__init__()

        self.flatten = nn.Flatten() # 将 [N, C, H, W] 展平为 [N, C*H*W]
                                    # 如果你的数据加载器已经展平了数据，可以注释掉这行，
                                    # 并确保 input_size 正确传入。

        # 定义网络的层
        # 考虑到你的 data_loader 输出是 [batch_size, 1, 28, 28]，
        # nn.Flatten() 会将其转换为 [batch_size, 784]。
        # 因此，第一个线性层的输入应该是 1*28*28 = 784。

        self.fc1 = nn.Linear(input_size, hidden_size1) # 第一个全连接层
        self.relu1 = nn.ReLU()                         # 第一个 ReLU 激活函数

        self.fc2 = nn.Linear(hidden_size1, hidden_size2) # 第二个全连接层
        self.relu2 = nn.ReLU()                           # 第二个 ReLU 激活函数

        self.fc3 = nn.Linear(hidden_size2, num_classes)  # 输出层 (全连接层)
        # 注意：CrossEntropyLoss 期望原始的 logits 作为输入，它内部会应用 softmax。
        # 所以这里不需要显式添加 Softmax 层。

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状通常为 [batch_size, 1, 28, 28] (如果使用 Flatten 层)
                              或 [batch_size, 784] (如果数据已展平)。
        返回:
            torch.Tensor: 模型的输出 (logits)，形状为 [batch_size, num_classes]。
        """
        # 如果 self.flatten 被定义 (即数据加载器未展平数据)
        x = self.flatten(x)

        # 通过各个层传递数据
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    # 这个 __main__ 块可以用来测试你的模型定义是否正确

    # --- 测试 FCNN (模型内部包含 Flatten) ---
    print("--- 测试 FCNN (模型内部包含 Flatten) ---")
    # 模拟输入数据 (一个批次的 MNIST 图像)
    # 假设 batch_size 为 64，图像为 1x28x28
    dummy_input_flattened_by_model = torch.randn(64, 1, 28, 28)

    # 实例化模型
    # 使用默认的隐藏层大小和类别数 (num_classes=10)
    model = FCNN() # num_classes 默认为 10
    print("模型结构:")
    print(model)

    # 将模拟输入传递给模型
    try:
        output = model(dummy_input_flattened_by_model)
        print(f"\n模拟输入形状 (模型自行展平): {dummy_input_flattened_by_model.shape}")
        print(f"模型输出形状: {output.shape}") # 应该输出 torch.Size([64, 10])
        # 使用模型定义的类别数进行断言
        assert output.shape == (64, model.fc3.out_features), f"输出形状不正确! 期望: (64, {model.fc3.out_features}), 得到: {output.shape}"
        print("模型前向传播测试通过 (使用模型内部Flatten)！")
    except Exception as e:
        print(f"模型前向传播测试失败 (使用模型内部Flatten): {e}")


    # --- 测试 FCNNPreFlattened (假设输入已预先展平) ---
    print("\n--- 测试 FCNNPreFlattened (假设输入已预先展平) ---")
    class FCNNPreFlattened(nn.Module): # 保持这个辅助类定义在测试块内
        def __init__(self, input_size: int = 784, hidden_size1: int = 512, hidden_size2: int = 256, num_classes: int = 10):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size2, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            return x

    dummy_input_pre_flattened = torch.randn(64, 784) # 输入已展平
    num_classes_for_test = 10 # 为这个测试定义类别数
    model_pre_flattened = FCNNPreFlattened(input_size=784, num_classes=num_classes_for_test)
    print("模型结构 (假设输入已展平):")
    print(model_pre_flattened)
    try:
        output_pre_flattened = model_pre_flattened(dummy_input_pre_flattened)
        print(f"\n模拟输入形状 (预展平): {dummy_input_pre_flattened.shape}")
        print(f"模型输出形状: {output_pre_flattened.shape}") # 应该输出 torch.Size([64, 10])
        assert output_pre_flattened.shape == (64, num_classes_for_test), f"预展平输入的输出形状不正确! 期望: (64, {num_classes_for_test}), 得到: {output_pre_flattened.shape}"
        print("模型前向传播测试通过 (假设输入已展平)！")
    except Exception as e:
        print(f"模型前向传播测试失败 (假设输入已展平): {e}")