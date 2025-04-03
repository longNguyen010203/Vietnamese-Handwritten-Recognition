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
x = Conv2D(64, (3, 3), padding='same')(inputs)   
x = MaxPool2D(pool_size=3, strides=3)(x)        
x = Activation('relu')(x)                       
x_1 = x  # Lưu lại đầu ra của Block 1 để sử dụng nếu cần

# Block 2: Tăng số filters
x = Conv2D(128, (3, 3), padding='same')(x)      
x = MaxPool2D(pool_size=3, strides=3)(x)         
x = Activation('relu')(x)                       
x_2 = x  # Lưu lại đầu ra của Block 2

# Block 3: Thêm Batch Normalization để ổn định quá trình huấn luyện
x = Conv2D(256, (3, 3), padding='same')(x)       
x = BatchNormalization()(x)                    
x = Activation('relu')(x)                  
x_3 = x  # Lưu lại đầu ra của Block 3

# Block 4: Residual Connection (Kết nối tắt)
x = Conv2D(256, (3, 3), padding='same')(x)  
x = BatchNormalization()(x)                      
x = Add()([x, x_3])  
x = Activation('relu')(x)                         
x_4 = x  # Lưu lại đầu ra của Block 4

# Block 5: Tăng số lượng filters
x = Conv2D(512, (3, 3), padding='same')(x)       
x = BatchNormalization()(x)                     
x = Activation('relu')(x)                         
x_5 = x  # Lưu lại đầu ra của Block 5

# Block 6: Residual Connection với 512 filters
x = Conv2D(512, (3, 3), padding='same')(x)       
x = BatchNormalization()(x)                     
x = Add()([x, x_5])  
x = Activation('relu')(x)                       

# Block 7: Lớp tích chập cuối cùng với 1024 filters
x = Conv2D(1024, (3, 3), padding='same')(x)      
x = BatchNormalization()(x)                    
x = MaxPool2D(pool_size=(3, 1))(x)             
x = Activation('relu')(x)                       

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
model.load_weights('./weight_model/model_weights.hdf5')
