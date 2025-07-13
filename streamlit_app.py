import streamlit as st
import numpy as np
import cv2
from inference_pipeline import detect_and_read_number_plate_img
from PIL import Image
import pandas as pd
import io
import tempfile
from video_preprocess import process_video

st.set_page_config(
    page_title="Automatic Number Plate Recognition",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- HEADER ---
st.markdown("""
    <div style='background: linear-gradient(90deg, #e3f2fd 0%, #ffffff 100%); border-radius: 18px; padding: 2.2em 1em 1.2em 1em; margin-bottom: 2em; box-shadow: 0 2px 12px rgba(60,60,60,0.06);'>
        <div style='text-align:center;'>
            <span style='font-family: "Segoe UI", "Roboto", "Arial", sans-serif; font-size:2.7rem; font-weight:900; color:#0d47a1; letter-spacing:-1px;'>Automatic Number Plate Recognition <span style="color:#3949ab;">(ANPR)</span></span>
            <div style='height: 4px; width: 80px; background: linear-gradient(90deg, #3949ab 0%, #e3f2fd 100%); margin: 0.7em auto 1.2em auto; border-radius: 2px;'></div>
            <span style='display:inline-block; font-size:1.18rem; color:#222; font-weight:400; max-width: 600px;'>Upload a vehicle image. The app will detect license plates and extract the license number(s) using <b>YOLOv8</b> and <b>OCR</b>.</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- UPLOAD AREA ---
st.markdown("<div style='margin-bottom:1.5em;'></div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Choose image(s) (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload one or more clear images of vehicles with visible license plates.",
    accept_multiple_files=True
)

uploaded_video = st.file_uploader(
    "Or upload a video (MP4, AVI, MOV)",
    type=["mp4", "avi", "mov"],
    help="Upload a video of a vehicle with visible license plates."
)

results = []

if uploaded_video is not None:
    st.markdown("<div style='margin-bottom:1.5em;'></div>", unsafe_allow_html=True)
    st.video(uploaded_video)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_vid:
        tmp_vid.write(uploaded_video.read())
        video_path = tmp_vid.name
    with st.spinner("Processing video for license plate detection and OCR..."):
        video_results = process_video(video_path, frame_interval=30, save_annotated=True)
    st.success(f"Processed {len(video_results)} frames.")
    # Prepare results for table and CSV
    video_table = []
    for res in video_results:
        if res['plates']:
            for plate in res['plates']:
                video_table.append({"Frame": res['frame'], "Plate": plate})
        else:
            video_table.append({"Frame": res['frame'], "Plate": ""})
    if video_table:
        import pandas as pd
        df_video = pd.DataFrame(video_table)
        st.markdown("<div style='font-size:1.15rem; font-weight:600; color:#1a237e; margin-bottom:0.5em;'>Detected Plates per Frame</div>", unsafe_allow_html=True)
        st.dataframe(df_video, use_container_width=True, hide_index=True)
        csv_video = df_video.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Video Results as CSV",
            data=csv_video,
            file_name="anpr_video_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    # Show up to 3 annotated frames as preview
    annotated_paths = [res['annotated_path'] for res in video_results if res['annotated_path']][:3]
    if annotated_paths:
        st.markdown("<div style='margin-top:1em; font-weight:500;'>Sample Annotated Frames:</div>", unsafe_allow_html=True)
        cols = st.columns(len(annotated_paths))
        for i, path in enumerate(annotated_paths):
            import cv2
            img = cv2.imread(path)
            cols[i].image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Frame {video_results[i]['frame']}")

if uploaded_files:
    st.markdown("<div style='margin-bottom:1.5em;'></div>", unsafe_allow_html=True)
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        with st.expander(f"Image: {uploaded_file.name}", expanded=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img, use_container_width=True)
            with st.spinner(f"Detecting and reading license plate(s) in {uploaded_file.name}..."):
                annotated_img, detected_texts = detect_and_read_number_plate_img(img.copy())
            with col2:
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Annotated Result")
            st.markdown("<b>Detected Plate(s):</b>", unsafe_allow_html=True)
            if detected_texts:
                for i, text in enumerate(detected_texts, 1):
                    st.markdown(f"<div style='background:#e3f2fd; border-radius:8px; padding:0.7em 1em; margin-bottom:0.5em; font-size:1.1rem; color:#0d47a1; font-weight:500;'>Plate {i}: <span style='font-size:1.2rem; font-weight:700;'>{text if text else 'No text detected'}</span></div>", unsafe_allow_html=True)
                    results.append({"Image": uploaded_file.name, "Plate": text})
            else:
                st.warning("No license plates detected.")
                results.append({"Image": uploaded_file.name, "Plate": ""})

    # --- SUMMARY TABLE & CSV ---
    if results:
        st.markdown("<div style='margin-top:2em;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:1.3rem; font-weight:600; color:#1a237e; margin-bottom:0.5em;'>Summary of All Results</div>", unsafe_allow_html=True)
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.markdown("<div style='margin-bottom:0.7em;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv,
            file_name="anpr_results.csv",
            mime="text/csv",
            use_container_width=True
        )

st.markdown("---")
st.caption("Made with ❤️ using Streamlit, YOLOv8, and Tesseract OCR.")
st.markdown("<div style='text-align:center; color: gray; font-size: 1rem; margin-top: 1em;'>Developed by <b>Mayur Bhagat</b></div>", unsafe_allow_html=True) 