# Automatic Number Plate Recognition (ANPR)

A professional, user-friendly web application for automatic number plate recognition using YOLOv8 and Tesseract OCR. Built with Streamlit, this app allows users to upload images or videos of vehicles, detects license plates, extracts license numbers, and provides results in a clean, modern interface. Results can be exported as CSV for further analysis.

---

## 🚀 Features

- **Image Upload:** Upload one or more vehicle images (JPG, JPEG, PNG) for license plate detection and OCR.
- **Video Upload:** Upload a video (MP4, AVI, MOV) for automatic frame extraction and license plate recognition.
- **YOLOv8 Detection:** Utilizes a custom-trained YOLOv8 model for accurate license plate localization.
- **Tesseract OCR:** Extracts license numbers from detected plates using Tesseract OCR.
- **Professional UI:** Clean, modern, and mobile-friendly interface with summary tables, previews, and CSV export.
- **CSV Export:** Download all detected license numbers (from images or video) as a CSV file.

---

## 🖥️ Demo

![App Screenshot](demo_screenshot.png)

---

## ⚙️ Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/automatic-number-plate-recognition.git
   cd automatic-number-plate-recognition
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract-OCR:**
   - **Windows:** [Download here](https://github.com/tesseract-ocr/tesseract/wiki)
   - **Linux:** `sudo apt-get install tesseract-ocr`
   - **Mac:** `brew install tesseract`
   - (Optional) Set the Tesseract path in `inference_pipeline.py` if not in your system PATH.

4. **Add your YOLOv8 model weights:**
   - Place your trained YOLOv8 weights (e.g., `best.pt`) in the `models/` directory.

5. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 📱 Usage Guide

1. **Open the app in your browser.**
2. **Upload images or a video** of vehicles with visible license plates.
3. **View results:**
   - For images: See detected plates, annotated previews, and a summary table.
   - For video: See detected plates per frame, sample annotated frames, and a summary table.
4. **Download results as CSV** for further analysis or record-keeping.

---

## 📦 Requirements
- Python 3.8+
- Streamlit
- OpenCV
- pytesseract
- ultralytics (YOLOv8)
- numpy
- pandas
- Tesseract-OCR (system package)

Install all Python dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🙏 Credits
- Developed by **Mayur Bhagat**
- Powered by [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- UI built with [Streamlit](https://streamlit.io/)

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
