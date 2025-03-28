# Import các thư viện cần thiết từ Keras và TensorFlow
import os
from tensorflow import keras
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Add, Activation
from keras.models import Model
import keras.backend as K

# Định nghĩa mô hình CRNN + LSTM nhận dạng ký tự

# Khởi tạo input với kích thước (118, 2167, 1) - ảnh xám đầu vào
inputs = Input(shape=(118, 2167, 1))

# Block 1: Lớp tích chập + MaxPooling
x = Conv2D(64, (3, 3), padding='same')(inputs)   # Convolutional layer với 64 filters, kernel size 3x3
x = MaxPool2D(pool_size=3, strides=3)(x)         # Max Pooling để giảm kích thước đặc trưng
x = Activation('relu')(x)                         # Hàm kích hoạt ReLU
x_1 = x  # Lưu lại đầu ra của Block 1 để sử dụng nếu cần

# Block 2: Tăng số filters
x = Conv2D(128, (3, 3), padding='same')(x)       # Convolutional layer với 128 filters
x = MaxPool2D(pool_size=3, strides=3)(x)         # Max Pooling
x = Activation('relu')(x)                         # ReLU activation
x_2 = x  # Lưu lại đầu ra của Block 2

# Block 3: Thêm Batch Normalization để ổn định quá trình huấn luyện
x = Conv2D(256, (3, 3), padding='same')(x)       # Convolutional layer với 256 filters
x = BatchNormalization()(x)                      # Chuẩn hóa Batch Normalization
x = Activation('relu')(x)                         # ReLU activation
x_3 = x  # Lưu lại đầu ra của Block 3

# Block 4: Residual Connection (Kết nối tắt)
x = Conv2D(256, (3, 3), padding='same')(x)       # Convolutional layer với 256 filters
x = BatchNormalization()(x)                      # Batch Normalization
x = Add()([x, x_3])  # Cộng đầu vào với đầu ra trước đó (Residual Connection)
x = Activation('relu')(x)                         # ReLU activation
x_4 = x  # Lưu lại đầu ra của Block 4

# Block 5: Tăng số lượng filters
x = Conv2D(512, (3, 3), padding='same')(x)       # Convolutional layer với 512 filters
x = BatchNormalization()(x)                      # Batch Normalization
x = Activation('relu')(x)                         # ReLU activation
x_5 = x  # Lưu lại đầu ra của Block 5

# Block 6: Residual Connection với 512 filters
x = Conv2D(512, (3, 3), padding='same')(x)       # Convolutional layer với 512 filters
x = BatchNormalization()(x)                      # Batch Normalization
x = Add()([x, x_5])  # Kết nối tắt để giữ lại thông tin
x = Activation('relu')(x)                         # ReLU activation

# Block 7: Lớp tích chập cuối cùng với 1024 filters
x = Conv2D(1024, (3, 3), padding='same')(x)      # Convolutional layer với 1024 filters
x = BatchNormalization()(x)                      # Batch Normalization
x = MaxPool2D(pool_size=(3, 1))(x)               # Max Pooling theo chiều cao
x = Activation('relu')(x)                         # ReLU activation

# Thêm một lớp MaxPooling nữa để giảm chiều cao ảnh
x = MaxPool2D(pool_size=(3, 1))(x)

# Loại bỏ chiều thứ nhất của tensor (batch size)
squeezed = Lambda(lambda x: K.squeeze(x, 1))(x)

# BiLSTM 1: LSTM hai chiều giúp nhận dạng ký tự tuần tự
blstm_1 = Bidirectional(
    LSTM(512, return_sequences=True, dropout=0.2))(squeezed)

# BiLSTM 2: Lớp LSTM tiếp theo để xử lý thông tin tuần tự
blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(blstm_1)

# Lớp đầu ra sử dụng softmax để phân loại ký tự (có 140 ký tự + 1 khoảng trắng)
outputs = Dense(140+1, activation='softmax')(blstm_2)

# Khởi tạo mô hình
model = Model(inputs, outputs)

# Tải trọng số của mô hình đã được huấn luyện trước đó
model.load_weights('./data/model_weights.hdf5')
