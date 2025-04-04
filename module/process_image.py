import cv2
import numpy as np
from matplotlib import pyplot as plt  # Import thư viện để vẽ biểu đồ

# Xử lý nhiều ảnh văn bản đầu vào
def process_multi(segments):
    """
    Hàm này xử lý nhiều ảnh đầu vào và trả về các ảnh đã xử lý dưới dạng numpy array.
    """
    copy_valid = []
    size = 0
    valid_imgs = []

    for img in segments:
        pre_img = process_image_mul(img)  # Xử lý từng ảnh
        valid_imgs.append(pre_img)  # Lưu ảnh đã xử lý vào danh sách
        size += 1

    copy_valid = valid_imgs.copy()  # Sao chép danh sách ảnh
    valid_imgs = np.array(valid_imgs)  # Chuyển thành numpy array

    return valid_imgs, copy_valid, size


# Xử lý một ảnh văn bản
def process_image_mul(cv2_img):
    """
    Tiền xử lý một ảnh đơn lẻ để chuẩn bị cho OCR.
    """
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang ảnh xám

    height = 118
    width = 2167

    img = cv2.bilateralFilter(img, 9, 80, 80)  # Lọc nhiễu bằng Bilateral Filter

    # Chuyển đổi sang ảnh nhị phân bằng Adaptive Threshold
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    # Chuẩn hóa kích thước ảnh bằng padding
    img_pad = padding_image(img, width, height)

    # Resize ảnh theo chiều cao cố định
    img = cv2.resize(img_pad, (int(118/height*width), 118))

    # Padding lại theo chiều rộng cố định
    img = np.pad(img, ((0, 0), (0, 2167-width)), 'median')

    # Áp dụng phép giãn ảnh (Dilation) để làm nổi bật đường biên của ký tự
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    img = np.expand_dims(img, axis=2)  # Mở rộng chiều dữ liệu để phù hợp với mô hình
    img = img / 255.  # Chuẩn hóa pixel về khoảng [0,1]

    return img


# Hàm xử lý ảnh khác, không sử dụng Bilateral Filter
def process_image(img_file):
    """
    Hàm này xử lý ảnh theo cách khác nhưng không dùng Bilateral Filter.
    """
    img = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám

    height = 118
    width = 2122

    img = cv2.resize(img, (int(118/height*width), 118))  # Resize theo chiều cao chuẩn
    img = np.pad(img, ((0, 0), (0, 2167-width)), 'median')  # Padding ảnh
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Làm mờ ảnh để giảm nhiễu

    # Chuyển đổi sang ảnh nhị phân
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    img = np.expand_dims(img, axis=2)  # Mở rộng chiều dữ liệu
    img = img / 255.  # Chuẩn hóa pixel

    return img


# Load ảnh gốc từ đường dẫn
def load_original_img(path):
    return cv2.imread(path)


# Chuyển đổi ảnh thành định dạng numpy array để sử dụng cho mô hình
def convert_img_to_input(img_file):
    """
    Hàm này nhận vào một ảnh và chuyển đổi nó thành định dạng numpy array để làm đầu vào cho mô hình.
    """
    valid_img = []
    valid_img.append(img_file)
    valid_img = np.array(valid_img)  # Chuyển thành numpy array
    return valid_img


# Hàm padding ảnh để chuẩn hóa kích thước
def padding_image(image, width, height):
    """
    Thêm viền đen vào ảnh để đưa về kích thước chuẩn.
    """
    h, w = image.shape[:2]  # Lấy kích thước ảnh
    color = [0, 0, 0]  # Màu đen

    if (h < height and w < width):
        # Nếu ảnh nhỏ hơn kích thước chuẩn, thêm viền vào hai bên
        top, bottom = int((height - h)/2), int((height - h)/2)
        left, right = 0, int((width - w)/2)
        new_img = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    else:
        if (h > height):  # Nếu chiều cao lớn hơn chuẩn, thêm viền trái/phải
            left, right = 0, int((width*h)/height)
            new_img = cv2.copyMakeBorder(
                image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=color)
        if (w > width):  # Nếu chiều rộng lớn hơn chuẩn, thêm viền trên/dưới
            top, bottom = int(((height*w)/width)/2), int(((height*w)/width)/2)
            new_img = cv2.copyMakeBorder(
                image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=color)

    return new_img


# Cắt ảnh theo kích thước mong muốn
def crop_image(image, width, height):
    """
    Cắt ảnh theo kích thước width x height từ tâm ảnh.
    """
    h, w = image.shape[:2]

    if (h > height and w > width):
        # Tính toán điểm bắt đầu để cắt ảnh từ trung tâm
        startx = w // 2 - (width // 2)
        starty = h // 2 - (height // 2)
        return image[starty:starty + height, startx:startx + width]
    else:
        return image  # Nếu ảnh nhỏ hơn kích thước mong muốn, giữ nguyên


# Áp dụng Erosion hoặc Dilation lên ảnh
def erosion_dilation_image(image, kernel_size, isErosion):
    """
    Thực hiện phép co ảnh (Erosion) hoặc giãn ảnh (Dilation) để xử lý văn bản.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if (isErosion == True):
        img = cv2.erode(image, kernel, iterations=1)  # Thực hiện phép co ảnh (Erosion)
    else:
        img = cv2.dilate(image, kernel, iterations=1)  # Thực hiện phép giãn ảnh (Dilation)

    return img
