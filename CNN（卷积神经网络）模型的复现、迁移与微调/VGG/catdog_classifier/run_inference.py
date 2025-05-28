# run_inference.py

from torchvision import transforms
from predictor import ImagePredictor  # 从我们刚创建的 predictor.py 导入
import os  # 用于检查路径是否存在

if __name__ == "__main__":
    # 1. 定义模型检查点文件的路径
    model_checkpoint = "./checkpoints/catdog_vgg16.pth"  # 指向您训练好的模型权重

    # 2. 定义图像预处理流程 (必须与训练时完全一致!)
    # 如果训练时有 Normalization，这里也必须加上
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 例如，如果训练时用了ImageNet的均值和标准差:
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. (可选但推荐) 定义类别名称
    # 顺序必须与模型训练时的类别索引对应。
    # 例如，如果 'cat' 对应索引0，'dog' 对应索引1
    class_names_list = ['cat', 'dog']
    # class_names_list = None  # 如果设为None，预测器会使用通用名称如 '类别_0', '类别_1'
    # 或者您可以之后调用 predictor.set_class_names_from_train_dir()

    # 4. 实例化 ImagePredictor
    try:
        predictor = ImagePredictor(
            model_checkpoint_path=model_checkpoint,
            image_transform=inference_transform,
            class_names=class_names_list
            # device 参数可以不传，会自动检测 CUDA 或 CPU
        )
    except Exception as e:
        print(f"初始化预测器失败: {e}")
        exit()  # 如果初始化失败，则退出程序

    # (可选步骤) 如果初始化时未提供 class_names_list，或者想通过训练目录结构推断：
    if predictor.class_names and "类别_0" in predictor.class_names[0]:  # 检查是否是通用名称
        train_data_directory = "./data/train"  # 指向您原始的训练数据目录
        if os.path.exists(train_data_directory):
            predictor.set_class_names_from_train_dir(train_data_directory)
        else:
            print(f"警告: 用于推断类别名称的训练目录 '{train_data_directory}' 不存在。")
    print(f"最终使用的类别名称进行预测: {predictor.class_names}")

    # 5. 指定包含新图片的目录
    # 您之前提供的路径是 'catdog_classifier/data/infer_set'
    # 假设 run_inference.py 在 'catdog_classifier' 目录下，则相对路径是 './data/infer_set'
    images_to_predict_dir = "./data/infer_set"

    # 6. 对目录中的图片进行预测
    if os.path.exists(images_to_predict_dir):
        # show_images_on_console=True 会尝试用 matplotlib 显示每张图片和结果
        # 注意：这会使程序在每张图片后暂停，直到关闭图像窗口
        prediction_results = predictor.predict_directory(images_to_predict_dir, show_images_on_console=True)

        # (可选) 处理预测结果列表 prediction_results
        for result in prediction_results:
            print(f"详细结果: {result}")
    else:
        print(f"错误：待预测图片的目录 '{images_to_predict_dir}' 未找到。")

    # # 7. (可选) 预测单张图片示例
    # single_image_file = os.path.join(images_to_predict_dir, "some_image_name.jpg") # 替换为具体图片名
    # if os.path.exists(single_image_file):
    #     print(f"\n--- 预测单张图片: {single_image_file} ---")
    #     name, idx, conf = predictor.predict_image(single_image_file)
    #     if name is not None:
    #         print(f"  -> 预测结果: {name} (索引: {idx}, 置信度: {conf*100:.2f}%)")
    # else:
    #     print(f"示例单张图片 '{single_image_file}' 未找到。")