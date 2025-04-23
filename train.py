import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

from config import Config
from model import ImprovedThymeModel, FocalLoss
from dataset import ThymeDataset


def train_model():
    print(f"使用设备: {Config.device}")

    print("\n加载数据集...")
    full_dataset = ThymeDataset(Config.data_root)

    all_indices = list(range(len(full_dataset)))
    all_labels = [full_dataset[i][1] for i in all_indices]

    class_indices = {i: [] for i in range(Config.num_classes)}
    for idx, label in zip(all_indices, all_labels):
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    np.random.seed(42)

    for class_id, indices in class_indices.items():
        indices = np.random.permutation(indices).tolist()

        n_train = int(len(indices) * 0.7)
        n_val = int(len(indices) * 0.15)
        n_test = len(indices) - n_train - n_val

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])


    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print(f"\n数据集划分:")
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本 (约占总数据集的 {len(test_indices) / len(full_dataset) * 100:.1f}%)")

    for class_id in range(Config.num_classes):
        n_train = sum(1 for i in train_indices if all_labels[i] == class_id)
        n_val = sum(1 for i in val_indices if all_labels[i] == class_id)
        n_test = sum(1 for i in test_indices if all_labels[i] == class_id)
        total = n_train + n_val + n_test

        print(f"类别 {class_id}: 训练集 {n_train}/{total} ({n_train / total * 100:.1f}%), "
              f"验证集 {n_val}/{total} ({n_val / total * 100:.1f}%), "
              f"测试集 {n_test}/{total} ({n_test / total * 100:.1f}%)")

  
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True
    )


    print("\n初始化模型...")
    model = ImprovedThymeModel(num_classes=Config.num_classes).to(Config.device)
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=Config.num_epochs // 3,
        T_mult=2,
        eta_min=Config.min_lr
    )


    scaler = GradScaler(enabled=True if Config.device == 'cuda' else False)


    best_val_acc = 0
    patience_counter = 0


    with open(os.path.join(Config.model_save_path, 'experiment_config.txt'), 'w') as f:
        f.write(f"实验: 完整ImprovedThymeModel模型 (4类)\n")
        f.write(f"数据集划分: 所有类别统一使用70%训练集、15%验证集、15%测试集\n")
        f.write(f"批次大小: {Config.batch_size}\n")
        f.write(f"学习率: {Config.learning_rate}\n")
        f.write(f"权重衰减: {Config.weight_decay}\n")
        f.write(f"训练轮数: {Config.num_epochs}\n")

    print("\n开始训练...")
    for epoch in range(Config.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.num_epochs}")
        for inputs, targets in pbar:
            inputs = inputs.to(Config.device)
            targets = targets.to(Config.device)

            optimizer.zero_grad()


            with autocast(enabled=True if Config.device == 'cuda' else False):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })

        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating"):
                inputs = inputs.to(Config.device)
                targets = targets.to(Config.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(f'\nEpoch {epoch + 1}/{Config.num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        print(f'Learning rate: {current_lr:.6f}')

        with open(os.path.join(Config.model_save_path, 'training_log.txt'), 'a') as f:
            f.write(f'Epoch {epoch + 1}/{Config.num_epochs}, ')
            f.write(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, ')
            f.write(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, ')
            f.write(f'LR: {current_lr:.6f}\n')

        if val_acc > best_val_acc:
            print(f"验证准确率提升: {best_val_acc:.2f}% -> {val_acc:.2f}%")
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, os.path.join(Config.model_save_path, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.patience:
                print(f'\n早停：{Config.patience} 轮未见改善')
                break


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    import datetime

    start_time = datetime.datetime.now()
    with open(os.path.join(Config.model_save_path, 'experiment_info.txt'), 'w') as f:
        f.write(f"实验开始时间: {start_time}\n")
        f.write(f"实验: 完整ImprovedThymeModel模型 (4类)\n")
        f.write(f"数据集划分: 所有类别统一使用70%训练集、15%验证集、15%测试集\n")

    try:
        train_model()
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        with open(os.path.join(Config.model_save_path, 'experiment_info.txt'), 'a') as f:
            f.write(f"实验结束时间: {end_time}\n")
            f.write(f"总耗时: {duration}\n")
    except Exception as e:
        import traceback

        with open(os.path.join(Config.model_save_path, 'error_log.txt'), 'w') as f:
            f.write(f"训练过程中出现错误:\n{str(e)}\n\n")
            f.write(traceback.format_exc())
