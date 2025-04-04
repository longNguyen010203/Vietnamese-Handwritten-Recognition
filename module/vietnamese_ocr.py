# Import các thư viện cần thiết từ TensorFlow và Keras
from tensorflow import keras
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Add, Activation
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import numpy as np

# Import mô hình CRNN từ module riêng
import module.crnn_model as crnn_model

# Số lượng ảnh được dự đoán tại mỗi lần nhận diện
NO_PREDICTS = 1  # Dự đoán 1 ảnh một lần
OFFSET = 0  # Dịch chỉ mục ảnh đầu vào nếu cần

# Danh sách các ký tự có thể nhận dạng được bởi mô hình
char_list = [' ', '#', "'", '(', ')', '+', ',', '-', '.',
             '/', '0', '1', '2', '3', '4', '5', '6', '7',
             '8', '9', ':', 'A', 'B', 'C', 'D', 'E', 'F',
             'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
             'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
             'Y', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i',
             'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
             't', 'u', 'v', 'w', 'x', 'y', 'z', 'Â', 'Ê',
             'Ô', 'à', 'á', 'â', 'ã', 'è', 'é', 'ê', 'ì',
             'í', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'ă',
             'Đ', 'đ', 'ĩ', 'ũ', 'Ơ', 'ơ', 'ư', 'ạ', 'ả',
             'ấ', 'ầ', 'ẩ', 'ậ', 'ắ', 'ằ', 'ẵ', 'ặ', 'ẻ',
             'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ỉ', 'ị', 'ọ',
             'ỏ', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ', 'ờ', 'ở',
             'ỡ', 'ợ', 'ụ', 'ủ', 'Ứ', 'ứ', 'ừ', 'ử', 'ữ',
             'ự', 'ỳ', 'ỵ', 'ỷ', 'ỹ']

# HÀM DỰ ĐOÁN KÝ TỰ TRÊN ẢNH ĐƠN LẺ

def prediction_ocr(valid_img):
    """
    Dự đoán ký tự từ ảnh đầu vào duy nhất.
    :param valid_img: Mảng numpy chứa ảnh đã xử lý trước
    :return: Chuỗi ký tự dự đoán được
    """
    # Dự đoán bằng mô hình CRNN đã tải
    prediction = crnn_model.model.predict(valid_img[OFFSET:OFFSET+NO_PREDICTS])

    # Giải mã kết quả bằng thuật toán CTC (Connectionist Temporal Classification)
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1], greedy=True)[0][0])

    # Xây dựng chuỗi kết quả
    pred = ""
    for x in out:
        for p in x:
            if int(p) != -1:  # -1 là giá trị "blank" trong CTC
                pred += char_list[int(p)]

    return pred  # Trả về chuỗi ký tự nhận diện được


# HÀM DỰ ĐOÁN KÝ TỰ TRÊN NHIỀU ẢNH

def prediction_ocr_multi(valid_img, SIZE):
    """
    Dự đoán ký tự từ nhiều ảnh đầu vào.
    :param valid_img: Mảng numpy chứa nhiều ảnh đã xử lý trước
    :param SIZE: Số lượng ảnh cần nhận diện
    :return: Chuỗi kết quả nhận diện của từng ảnh (mỗi ảnh trên một dòng)
    """
    # Dự đoán kết quả trên toàn bộ ảnh đầu vào
    prediction = crnn_model.model.predict(valid_img[OFFSET:OFFSET+SIZE])

    # Giải mã kết quả dự đoán bằng thuật toán CTC
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1], greedy=True)[0][0])

    all_predictions = []  # Danh sách lưu kết quả của từng ảnh

    # Duyệt qua từng kết quả dự đoán
    for x in out:
        pred = ""
        for p in x:
            if int(p) != -1:  # -1 là giá trị "blank" trong CTC
                pred += char_list[int(p)]
        all_predictions.append(pred)  # Thêm kết quả vào danh sách

    return '\n'.join(all_predictions)  # Ghép kết quả thành chuỗi nhiều dòng
