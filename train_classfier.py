import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# 1. 加载 CSV 数据
def load_data(csv_path):
    # 读取 CSV 文件
    data = pd.read_csv(csv_path)

    # 检查是否有 NaN 值
    if data.isnull().values.any():
        print("Warning: Training data contains NaN values.")
        # 处理 NaN 值
        data = data.dropna()

    # 提取 3D 特征（所有关键点的 x, y, z）
    features_3d = data.filter(regex='_x$|_y$|_z$').values
    print(features_3d, features_3d.shape)

    # 提取标签
    labels = data['label'].values

    return features_3d, labels


# 2. 数据预处理
def preprocess_data(features, labels):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 保存模型
    joblib.dump(scaler, 'saved_model/scaler.pkl')

    return X_train_scaled, X_test_scaled, y_train, y_test


# 3. 训练模型
def train_model(X_train, y_train, model_type='SVM'):
    if model_type == 'SVM':
        model = SVC(kernel='rbf', probability=True)
    elif model_type == 'KNN':
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError("Unsupported model type. Choose 'SVM' or 'KNN'.")

    model.fit(X_train, y_train)
    return model


# 4. 模型评估
def evaluate_model(model, X_test, y_test):
    # 预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2%}")

    # 显示混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Gesture", "Release"], yticklabels=["Gesture", "Release"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


# 5. 主函数
def main(csv_path, model_type='SVM'):
    # 加载数据
    features, labels = load_data(csv_path)

    # 数据预处理
    X_train, X_test, y_train, y_test = preprocess_data(features, labels)

    # 训练模型
    model = train_model(X_train, y_train, model_type=model_type)

    # 模型评估
    evaluate_model(model, X_test, y_test)

    # 保存模型
    joblib.dump(model, 'saved_model/model.pkl')


# 6. 运行主函数
if __name__ == "__main__":
    csv_path = "hand_data.csv"
    model_type = "SVM"  # 可选：'SVM' 或 'KNN'
    main(csv_path, model_type)
