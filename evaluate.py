import os
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

from config import Config
from model import ImprovedThymeModel
from dataset import ThymeDataset


def evaluate_specific_model(model_path, output_dir):
    """评估特定模型并将结果保存在指定目录中"""

    # 打印评估信息
    model_filename = os.path.basename(model_path)
    print(f"正在评估模型: {model_filename}")
    print(f"结果将保存到: {output_dir}")

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 加载完整数据集
    print("加载数据集...")
    full_dataset = ThymeDataset(Config.data_root)

    # 获取所有样本的索引和标签
    all_indices = list(range(len(full_dataset)))
    all_labels = [full_dataset[i][1] for i in all_indices]

    # 按类别分组索引
    class_indices = {i: [] for i in range(Config.num_classes)}
    for idx, label in zip(all_indices, all_labels):
        class_indices[label].append(idx)

    # 划分训练集、验证集和测试集的索引
    train_indices = []
    val_indices = []
    test_indices = []

    # 设置随机种子保证可重复性
    np.random.seed(42)

    for class_id, indices in class_indices.items():
        # 打乱该类别的索引
        indices = np.random.permutation(indices).tolist()

        # 所有类别统一使用70-15-15的划分比例
        n_train = int(len(indices) * 0.7)
        n_val = int(len(indices) * 0.15)
        n_test = len(indices) - n_train - n_val

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    # 创建子数据集
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本 (约占总数据集的 {len(test_indices) / len(full_dataset) * 100:.1f}%)")

    # 输出每个类别在各数据集中的分布
    for class_id in range(Config.num_classes):
        n_train = sum(1 for i in train_indices if all_labels[i] == class_id)
        n_val = sum(1 for i in val_indices if all_labels[i] == class_id)
        n_test = sum(1 for i in test_indices if all_labels[i] == class_id)
        total = n_train + n_val + n_test

        print(f"类别 {class_id}: 训练集 {n_train}/{total} ({n_train / total * 100:.1f}%), "
              f"验证集 {n_val}/{total} ({n_val / total * 100:.1f}%), "
              f"测试集 {n_test}/{total} ({n_test / total * 100:.1f}%)")

    # 创建测试集的DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    # 加载指定的模型
    print(f"加载模型 {model_path}...")
    model = ImprovedThymeModel(num_classes=Config.num_classes)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.device)
    model.eval()

    # 类别映射
    stage_names = {
        0: 'Green-up',
        1: 'Early flowering',
        2: 'Peak flowering',
        3: 'Fruiting'
    }

    # 存储预测结果
    all_preds = []
    all_labels = []
    all_probs = []

    print("在测试集上评估模型...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算各项指标
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    accuracy = accuracy_score(all_labels, all_preds)

    # 创建详细的指标DataFrame
    metrics_data = []
    for i in range(Config.num_classes):
        metrics_data.append({
            'Stage': stage_names[i],
            'Precision': f"{precision[i] * 100:.2f}%",
            'Recall': f"{recall[i] * 100:.2f}%",
            'F1-Score': f"{f1[i] * 100:.2f}%",
            'Support': np.sum(all_labels == i)
        })

    metrics_df = pd.DataFrame(metrics_data)

    # 添加总体准确率
    metrics_df.loc[len(metrics_df)] = ['Overall Accuracy',
                                       f"{np.mean(precision) * 100:.2f}%",
                                       f"{np.mean(recall) * 100:.2f}%",
                                       f"{np.mean(f1) * 100:.2f}%",
                                       len(all_labels)]

    # 提取模型名称（不含后缀）
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # 保存详细指标
    metrics_path = os.path.join(output_dir, f'{model_name}_detailed_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')

    # 打印结果
    print("\n详细指标:")
    print(metrics_df)
    print(f"\n整体准确率: {accuracy * 100:.2f}%")

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 使用系统默认的serif字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman',
                                  'New Century Schoolbook', 'Century Schoolbook L', 'Utopia',
                                  'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Times New Roman',
                                  'Times', 'Palatino', 'Charter', 'serif']

    plt.figure(figsize=(14, 10))

    # 设置混淆矩阵的显示样式
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[stage_names[i] for i in range(Config.num_classes)],
                yticklabels=[stage_names[i] for i in range(Config.num_classes)])

    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=16, labelpad=10)
    plt.ylabel('True Label', fontsize=16, labelpad=10)

    # 标签不倾斜
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(rotation=45, fontsize=14)

    # 调整布局以确保所有标签都可见
    plt.tight_layout()

    # 保存混淆矩阵图
    confusion_matrix_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')

    # 也保存一个PDF版本以保持最佳质量
    confusion_matrix_pdf_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.pdf')
    plt.savefig(confusion_matrix_pdf_path, format='pdf', bbox_inches='tight')
    plt.close()

    # 计算每个类别的具体指标
    class_metrics = {}
    for i in range(Config.num_classes):
        class_metrics[stage_names[i]] = {
            'Precision': precision[i],
            'Recall': recall[i],
            'F1-score': f1[i],
            'Support': np.sum(all_labels == i)
        }

    # 保存为Excel格式的详细报告
    with pd.ExcelWriter(os.path.join(output_dir, f'{model_name}_evaluation_report.xlsx')) as writer:
        # 保存整体指标
        overall_metrics = pd.DataFrame({
            'Metric': ['Accuracy', 'Macro Avg Precision', 'Macro Avg Recall', 'Macro Avg F1-score'],
            'Value': [
                f"{accuracy * 100:.2f}%",
                f"{np.mean(precision) * 100:.2f}%",
                f"{np.mean(recall) * 100:.2f}%",
                f"{np.mean(f1) * 100:.2f}%"
            ]
        })
        overall_metrics.to_excel(writer, sheet_name='Overall Metrics', index=False)

        # 保存每个类别的详细指标
        metrics_df.to_excel(writer, sheet_name='Class Metrics', index=False)

        # 保存混淆矩阵
        pd.DataFrame(cm,
                     columns=[stage_names[i] for i in range(Config.num_classes)],
                     index=[stage_names[i] for i in range(Config.num_classes)]).to_excel(writer,
                                                                                         sheet_name='Confusion Matrix')

        # 添加模型信息
        model_info = pd.DataFrame({
            'Info': ['Model Filename', 'Evaluation Date'],
            'Value': [model_filename, pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
        })
        model_info.to_excel(writer, sheet_name='Model Info', index=False)

    # 记录评估完成信息和数据集划分变更
    with open(os.path.join(output_dir, f'{model_name}_evaluation_results.txt'), 'w') as f:
        f.write(f"模型评估结果: {model_filename}\n")
        f.write(f"评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据集划分: 所有类别统一使用70%训练集、15%验证集、15%测试集\n\n")
        f.write(f"整体准确率: {accuracy * 100:.2f}%\n")
        f.write(f"宏平均精确率: {np.mean(precision) * 100:.2f}%\n")
        f.write(f"宏平均召回率: {np.mean(recall) * 100:.2f}%\n")
        f.write(f"宏平均F1分数: {np.mean(f1) * 100:.2f}%\n\n")

        f.write("每个类别的性能:\n")
        for i in range(Config.num_classes):
            f.write(
                f"{stage_names[i]}: 精确率={precision[i] * 100:.2f}%, 召回率={recall[i] * 100:.2f}%, F1={f1[i] * 100:.2f}%\n")

    print(f"\n评估结果已保存到: {output_dir}")
    print(f"详细指标: {model_name}_detailed_metrics.csv")
    print(f"混淆矩阵: {model_name}_confusion_matrix.png 和 {model_name}_confusion_matrix.pdf")
    print(f"完整报告: {model_name}_evaluation_report.xlsx")

    return metrics_df, cm


if __name__ == '__main__':
    # 指定要评估的模型路径和结果保存路径
    model_path = "/root/autodl-tmp/saved_models_4pheno/DApheno_Resnet111/model_test_acc92.48_epoch17.pth"
    output_dir = "/root/autodl-tmp/saved_models_4pheno/DApheno_Resnet111"

    # 评估指定模型
    evaluate_specific_model(model_path, output_dir)