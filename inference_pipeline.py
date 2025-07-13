import cv2
import pytesseract
from ultralytics import YOLO
import numpy as np
import re

# Optional: set Tesseract path if on Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLOv8 model
model = YOLO("models/best.pt")  # path to your YOLOv8 weights

def clean_plate_text(text):
    # Keep only alphanumeric characters (uppercase) and remove others
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def detect_and_read_number_plate_img(img):
    """
    Accepts a numpy array image, returns annotated image and list of detected texts.
    """
    results = model(img)[0]  # get first result (Batch size = 1)
    detected_texts = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = img[y1:y2, x1:x2]
        plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        plate_text = pytesseract.image_to_string(plate_gray, config="--psm 8")
        cleaned_text = clean_plate_text(plate_text.strip())
        detected_texts.append(cleaned_text)
        # Draw box and OCR result on original image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, cleaned_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return img, detected_texts


if __name__ == "__main__":
    test_image = "images/test.jpg"  # Change this to your test image
    img = cv2.imread(test_image)
    annotated_img, detected_texts = detect_and_read_number_plate_img(img)
    print("Detected Texts:", detected_texts)
    cv2.imshow("Annotated Image", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
