import cv2
import os
import tempfile
from inference_pipeline import detect_and_read_number_plate_img

def process_video(video_path, frame_interval=30, save_annotated=False):
    """
    Extract frames from video at every `frame_interval`, run detection+OCR, and return results.
    Args:
        video_path (str): Path to the video file.
        frame_interval (int): Process every Nth frame.
        save_annotated (bool): If True, save annotated frames to a temp directory.
    Returns:
        List of dicts: [{ 'frame': int, 'plates': [str], 'annotated_path': str or None }]
    """
    cap = cv2.VideoCapture(video_path)
    results = []
    frame_count = 0
    savedir = tempfile.mkdtemp(prefix="anpr_frames_") if save_annotated else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            annotated_img, detected_texts = detect_and_read_number_plate_img(frame.copy())
            annotated_path = None
            if save_annotated and savedir:
                annotated_path = os.path.join(savedir, f"frame_{frame_count}.jpg")
                cv2.imwrite(annotated_path, annotated_img)
            results.append({
                'frame': frame_count,
                'plates': detected_texts,
                'annotated_path': annotated_path
            })
        frame_count += 1
    cap.release()
    return results 