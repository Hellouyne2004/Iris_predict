import joblib
import sys
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
sys.stdout.reconfigure(encoding='utf-8')
# 1. Tải dữ liệu
iris = load_iris()
X, y = iris.data, iris.target

# Chia tập train/test để kiểm tra độ chính xác thực tế
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Tạo một Pipeline
# Pipeline giúp tự động hóa: Dữ liệu vào -> Chuẩn hóa (Scaler) -> Dự đoán (SVM)
pipeline = Pipeline([
    ('scaler', StandardScaler()),      # Bước 1: Chuẩn hóa dữ liệu
    ('svm', SVC())                     # Bước 2: Mô hình SVM
])

# 3. Thiết lập GridSearch để tìm tham số tốt nhất
param_grid = {
    'svm__C': [0.1, 1, 10, 100],              # Thử các độ phạt lỗi khác nhau
    'svm__gamma': [1, 0.1, 0.01, 0.001],      # Thử các hệ số kernel
    'svm__kernel': ['rbf', 'linear']          # Thử loại thuật toán
}

grid = GridSearchCV(pipeline, param_grid, cv=5, verbose=1)

# 4. Huấn luyện (Máy sẽ chạy thử tất cả các tổ hợp để tìm cái tốt nhất)
print("Đang huấn luyện và tìm tham số tối ưu...")
grid.fit(X_train, y_train)

# 5. Đánh giá kết quả
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nĐộ chính xác trên tập kiểm tra: {accuracy * 100:.2f}%")
print(f"Tham số tốt nhất tìm được: {grid.best_params_}")

# 6. Lưu model tốt nhất (bao gồm cả bước Scaler bên trong)
joblib.dump(best_model, 'iris_model.pkl')
print("Đã lưu model mới (đã tối ưu) thành công: iris_model.pkl")