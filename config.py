import torch
import os

class Config:
    # 数据集路径
    data_root = "/root/autodl-tmp/thyme_raw_data_4pheno"

    # 训练参数
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001

    # 模型参数
    num_classes = 4  # 4个生育期类别

    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据加载参数
    num_workers = 2
    prefetch_factor = 2

    # 保存模型的路径
    model_save_path = "/root/autodl-tmp/saved_models_4pheno/DApheno_Resnet111"
    os.makedirs(model_save_path, exist_ok=True)

    # 训练优化参数 - 新增
    mixed_precision = True  # 使用混合精度训练
    gradient_accumulation_steps = 1  # 梯度累积步数
    weight_decay = 0.01  # 权重衰减
    warmup_epochs = 5  # 预热轮数
    patience = 10  # 早停耐心值
    min_lr = 1e-6  # 最小学习率

    # 数据增强参数
    img_size = 224
    train_transforms_params = {
        'scale': (0.8, 1.0),
        'rotation': 30,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2
        }
    }