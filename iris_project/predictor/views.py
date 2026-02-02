import os
import joblib
import numpy as np
from django.shortcuts import render
from django.conf import settings

# Đường dẫn tới file model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'iris_model.pkl')

def predict_iris(request):
    prediction_text = ""
    
    # Khởi tạo giá trị mặc định là rỗng (để khi mới vào trang không bị lỗi)
    sepal_length = ''
    sepal_width = ''
    petal_length = ''
    petal_width = ''

    if request.method == 'POST':
        try:
            # 1. Lấy dữ liệu dạng chuỗi để hiển thị lại
            sepal_length = request.POST.get('sepal_length')
            sepal_width = request.POST.get('sepal_width')
            petal_length = request.POST.get('petal_length')
            petal_width = request.POST.get('petal_width')

            # 2. Chuyển sang float để tính toán
            # (Cần ép kiểu float ở đây để đưa vào model, nhưng biến lưu ở trên giữ nguyên string để in ra html)
            input_data = np.array([[
                float(sepal_length), 
                float(sepal_width), 
                float(petal_length), 
                float(petal_width)
            ]])

            # 3. Load model và dự đoán
            model = joblib.load(MODEL_PATH)
            prediction = model.predict(input_data)
            
            target_names = ['Setosa', 'Versicolor', 'Virginica']
            result_name = target_names[prediction[0]]
            
            prediction_text = f"Kết quả dự đoán: Hoa Iris {result_name}"

        except ValueError:
            prediction_text = "Vui lòng nhập đầy đủ số liệu hợp lệ."

    # 4. Đóng gói tất cả vào context để gửi ra HTML
    context = {
        'result': prediction_text,
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }

    return render(request, 'index.html', context)