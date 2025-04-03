import numpy as np
import cv2 
from PIL import Image  # Thư viện xử lý ảnh dạng PIL
import module.process_image as process_image  # Xử lý ảnh trước khi OCR
import module.vietnamese_ocr as vietnamese_ocr  # Mô hình OCR dùng CRNN + CTC
import module.crop_text_line as segments  # Cắt dòng văn bản từ ảnh

def prediction_ocr_crnn_ctc(img_model_input):
    """
    Nhận diện văn bản từ ảnh sử dụng mô hình CRNN + CTC.
    :param img_model_input: Ảnh đầu vào đã qua xử lý
    :return: Chuỗi văn bản nhận diện được
    """
    str_pred = vietnamese_ocr.prediction_ocr(img_model_input)
    print('Prediction:')
    print(str_pred)
    return str_pred

def prediction_multiline(img_model_input, size):
    """
    Nhận diện văn bản từ ảnh nhiều dòng bằng mô hình CRNN + CTC.
    :param img_model_input: Ảnh đầu vào
    :param size: Số dòng văn bản cần dự đoán
    :return: Chuỗi văn bản đã nhận diện
    """
    str_pred = vietnamese_ocr.prediction_ocr_multi(img_model_input, size)
    print('Prediction: ', str_pred)
    return str_pred

def test_prediction_mul(image_path):
    """
    Nhận diện văn bản từ ảnh có nhiều dòng bằng cách cắt từng dòng trước khi nhận diện.
    :param image_path: Đường dẫn ảnh đầu vào
    """
    all_predictions = []

    # Đọc ảnh gốc từ đường dẫn
    ori_img = cv2.imread(image_path)

    # Cắt từng dòng văn bản ra khỏi ảnh
    valid_img, arr = segments.segmentation_text_line(ori_img)

    # Dự đoán từng dòng bằng CRNN + CTC
    for img in arr:
        np_image = np.asarray(img)
        image_rgb = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        str_pred = vietnamese_ocr.prediction_ocr(image_pil)
        all_predictions.append(str_pred)

    # In toàn bộ văn bản đã nhận diện
    print('\n'.join(all_predictions))

# Gọi thử nghiệm nhận diện từ ảnh (có thể mở comment để chạy thử)
# if _name_ == "_main_":
#     test_prediction_mul('test_n1.jpg')