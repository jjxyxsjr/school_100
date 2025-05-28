# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models  # 用于加载预训练模型

from models.alexnet_model import AlexNet  # 导入我们自己定义的AlexNet

from data_loader import get_data_loaders
from train import train_one_epoch



def get_alexnet_model():
    """获取并初始化AlexNet模型，加载预训练权重并修改分类器"""
    model = AlexNet(num_classes=1000).to(config.device)

    try:
        pretrained_alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        pretrained_state_dict = pretrained_alexnet.state_dict()

        # 过滤掉最后一层分类器的权重
        new_state_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('classifier.6.')}
        model.load_state_dict(new_state_dict, strict=False)
        print("成功加载ImageNet预训练权重（除了分类器的最后一层）。")

    except Exception as e:
        print(f"无法加载预训练权重！错误: {e}. 将使用随机初始化的模型进行训练。")

    # 冻结特征提取层
    for param in model.features.parameters():
        param.requires_grad = False

    # 冻结分类器前几层
    for param in model.classifier.parameters():
        param.requires_grad = False

    # 修改分类器的最后一层以适应目标数据集的类别数量
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, config.num_classes).to(config.device)

    # 确保新替换的层是可训练的
    for param in model.classifier[6].parameters():
        param.requires_grad = True

    return model


def main():
    print(f"使用设备: {config.device}")

    # 服务器端模型（全局模型）
    global_model = get_alexnet_model()
    print("\n--- 全局 AlexNet 模型结构 (修改后) ---")
    print(global_model)

    # 准备服务器端评估数据 (使用完整测试集)
    server_test_loader = get_data_loaders(is_train=False, client_id=None)
    criterion = nn.CrossEntropyLoss()

    # 模拟客户端列表
    clients = []
    for i in range(config.num_clients):
        client_model = get_alexnet_model()  # 每个客户端开始时都有一个独立的模型副本
        client_train_loader = get_data_loaders(is_train=True, client_id=i)
        client_test_loader = get_data_loaders(is_train=False, client_id=i)  # 客户端本地评估，可选

        # 优化器只针对可训练的参数
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, client_model.parameters()),
                              lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)

        clients.append({
            'id': i,
            'model': client_model,
            'train_loader': client_train_loader,
            'test_loader': client_test_loader,  # 客户端本地测试集
            'optimizer': optimizer
        })

    print(f"\n--- 开始模拟联邦学习过程 (共 {config.num_epochs} 轮) ---")

    for global_round in range(1, config.num_epochs + 1):
        print(f"\n======== 联邦学习全局轮次: {global_round} ========")

        # 1. 服务器发送全局模型参数到所有客户端
        # 注意：这里为了简化，我们假设所有客户端都参与每一轮。
        # 在真实Flower中，可以选择部分客户端参与。
        global_model_state_dict = global_model.state_dict()

        client_updates = []  # 存储客户端发送回的参数更新

        for client_info in clients:
            client_id = client_info['id']
            client_model = client_info['model']
            client_train_loader = client_info['train_loader']
            client_optimizer = client_info['optimizer']

            print(f"客户端 {client_id}：接收全局模型，开始本地训练。")

            # 将全局模型的参数加载到客户端本地模型
            client_model.load_state_dict(global_model_state_dict)

            # 客户端本地训练
            for local_epoch in range(1, config.local_epochs + 1):
                print(f"客户端 {client_id} 本地训练 Epoch {local_epoch}/{config.local_epochs}")
                train_loss, train_acc = train_one_epoch(
                    client_model, client_train_loader, criterion, client_optimizer,
                    local_epoch, config.device
                )

            # 客户端发送更新后的参数（这里直接发送整个state_dict）
            client_updates.append(client_model.state_dict())
            print(f"客户端 {client_id}：本地训练完成，发送模型参数。")

        # 2. 服务器聚合客户端更新
        print("\n服务器：聚合客户端模型参数...")
        # 简单的平均聚合：将所有客户端的模型参数平均
        aggregated_state_dict = {}
        for key in global_model_state_dict.keys():
            aggregated_state_dict[key] = torch.stack([client_sd[key] for client_sd in client_updates]).mean(dim=0)

        # 将聚合后的参数加载回全局模型
        global_model.load_state_dict(aggregated_state_dict)
        print("服务器：模型参数聚合完成。")

        # 3. 服务器评估全局模型
        print("\n服务器：评估全局模型...")
        global_test_loss, global_test_acc = evaluate_model(global_model, server_test_loader, criterion, config.device)
        print(f"全局轮次 {global_round} 评估结果: "
              f"全局测试损失: {global_test_loss:.4f}, 全局测试准确率: {global_test_acc:.2f}%")

    print("\n--- 联邦学习模拟训练完成 ---")


if __name__ == '__main__':
    main()