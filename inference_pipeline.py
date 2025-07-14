import cv2
import pytesseract
from ultralytics import YOLO
import numpy as np
import re

# Optional: Windows-only Tesseract path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLOv8 model
model = YOLO("models/best.pt")  # Path to your trained YOLOv8 weights

def clean_plate_text(text):
    """Remove all non-alphanumeric characters and force uppercase."""
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def preprocess_plate(plate_img):
    """Enhance plate image for better OCR."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 2)
    return thresh

def detect_and_read_number_plate_img(img):
    """
    Takes a numpy array image, returns annotated image and list of detected texts.
    """
    results = model(img)[0]
    detected_texts = []

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = img[y1:y2, x1:x2]

        # Preprocess plate for OCR
        processed = preprocess_plate(plate_crop)

        # Run Tesseract OCR with whitelist & better config
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        raw_text = pytesseract.image_to_string(processed, config=custom_config)
        cleaned_text = clean_plate_text(raw_text.strip())

        detected_texts.append(cleaned_text)

        # Draw results
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, cleaned_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Optional: show cropped and processed plate
        # cv2.imshow(f"Plate {i+1}", plate_crop)
        # cv2.imshow(f"Processed {i+1}", processed)

    return img, detected_texts

if __name__ == "__main__":
    test_image_path = "images/test.jpg"
    img = cv2.imread(test_image_path)
    if img is None:
        print("Image not found:", test_image_path)
    else:
        annotated_img, detected_texts = detect_and_read_number_plate_img(img)
        print("🔍 Detected License Plate Texts:", detected_texts)

        cv2.imshow("Annotated Image", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
