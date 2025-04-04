import cv2
import numpy as np


def segmentation_text_line(image):
    """
    Hàm này nhận một ảnh đầu vào và thực hiện phân đoạn từng dòng văn bản trong ảnh.
    Trả về ảnh gốc với đường viền bao quanh từng dòng và danh sách các ảnh con chứa từng dòng văn bản.
    """

    img_clone = image.copy()  # Sao chép ảnh gốc để vẽ bounding box
    segments = []  # Danh sách chứa các đoạn văn bản đã được cắt ra

    # Chuyển đổi ảnh sang grayscale (ảnh xám)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Làm mờ ảnh bằng Gaussian Blur để giảm nhiễu
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Chuyển ảnh sang ảnh nhị phân (đen trắng) bằng Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    # Thực hiện phép co ảnh (Erosion) để loại bỏ nhiễu nhỏ
    kernel = np.ones((10, 1), np.uint8)  # Kernel dùng cho erosion
    ero = cv2.erode(thresh, kernel, iterations=1)

    # Thực hiện phép giãn ảnh (Dilation) để nối các phần của dòng chữ lại với nhau
    kernel = np.ones((5, 150), np.uint8)  # Kernel dùng cho dilation
    img_dilation = cv2.dilate(ero, kernel, iterations=1)

    # Tìm các đường viền (contours) trong ảnh sau khi giãn nở
    ctrs, hier = cv2.findContours(
        img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp các contour theo vị trí y (tức là theo chiều dọc, từ trên xuống dưới)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])

    # Định nghĩa ngưỡng tối thiểu cho chiều rộng và chiều cao của một dòng văn bản
    min_width = 50  # Giá trị tối thiểu cho chiều rộng dòng chữ
    min_height = 10  # Giá trị tối thiểu cho chiều cao dòng chữ

    # Duyệt qua từng contour đã sắp xếp
    for i, ctr in enumerate(sorted_ctrs):
        # Lấy tọa độ của contour (bounding box)
        x, y, w, h = cv2.boundingRect(ctr)

        # In ra kích thước của từng vùng được phát hiện
        print('segment no:', str(i), '[', w, '-', h, ']')

        # Chỉ giữ lại các vùng có kích thước đủ lớn để tránh nhiễu
        if (w > min_width and h > min_height):
            roi = image[y:y+h, x:x+w]  # Cắt vùng chứa văn bản từ ảnh gốc
            segments.append(roi)  # Thêm vào danh sách các đoạn văn bản

            # Vẽ hình chữ nhật bao quanh dòng văn bản trên ảnh gốc
            cv2.rectangle(img_clone, (x, y), (x + w, y + h), (90, 0, 255), 2)

    # Trả về ảnh gốc với các đường viền đã vẽ + danh sách các ảnh chứa từng dòng văn bản
    return img_clone, segments
