import cv2
import pytesseract
from ultralytics import YOLO
import numpy as np
import re
import easyocr

# Load YOLOv8 model
model = YOLO("models/best.pt")  # path to your YOLOv8 weights

# Optional: set Tesseract path (Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize EasyOCR reader
easyocr_reader = easyocr.Reader(['en'], gpu=False)

def clean_plate_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def preprocess_plate_strong(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    filtered = cv2.bilateralFilter(resized, 11, 17, 17)
    _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def is_valid_plate(text):
    # Optional: regex for Indian plate format (MH05DS8679)
    return re.match(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$', text) is not None

def ocr_tesseract(plate_img):
    processed = preprocess_plate_strong(plate_img)
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(processed, config=config)
    return clean_plate_text(text)

def ocr_easyocr(plate_img):
    result = easyocr_reader.readtext(plate_img)
    if result:
        text = result[0][1]
        return clean_plate_text(text)
    return ""

def detect_and_read_number_plate_img(img, use_easyocr=False):
    results = model(img)[0]
    detected_texts = []

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = img[y1:y2, x1:x2]

        # OCR with selected method
        if use_easyocr:
            raw_text = ocr_easyocr(plate_crop)
        else:
            raw_text = ocr_tesseract(plate_crop)

        if raw_text:
            detected_texts.append(raw_text)

            # Annotate results
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            color = (0, 0, 255) if not is_valid_plate(raw_text) else (255, 0, 0)
            cv2.putText(img, raw_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return img, detected_texts


if __name__ == "__main__":
    test_image_path = "images/test.jpg"  # Update path if needed
    img = cv2.imread(test_image_path)

    if img is None:
        print("Image not found:", test_image_path)
    else:
        # Toggle EasyOCR here
        annotated_img, detected_texts = detect_and_read_number_plate_img(img, use_easyocr=False)

        print("Detected Texts:", detected_texts)
        cv2.imshow("Annotated Result", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
