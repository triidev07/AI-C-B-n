from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # Sửa lỗi import

# Tạo đối tượng model YOLO với trọng số đã được huấn luyện
model = YOLO("yolov8n.pt")  # Sửa lỗi tên biến và dấu ngoặc đơn

# Mở camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Thiết lập độ rộng của khung hình
cap.set(4, 480)  # Thiết lập độ cao của khung hình

while True:
    ret, img = cap.read()  # Lấy khung hình từ camera
    if not ret:
        break

    # Dự đoán đối tượng trong hình ảnh
    results = model.predict(img)

    # Hiển thị kết quả dự đoán
    annotator = Annotator(img)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # Tọa độ của hộp (left, top, right, bottom)
            c = box.cls  # Nhãn của đối tượng
            annotator.box_label(b, model.names[int(c)])

    # Hiển thị hình ảnh với các hộp được chú thích
    img = annotator.result()
    img_resized = cv2.resize(img, (960, 540))
    cv2.imshow('YOLO V8 Detection', img_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Thoát khi nhấn phím 'q'
        break

cap.release()
cv2.destroyAllWindows()
