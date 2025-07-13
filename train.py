from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=4,
        workers=0,
        device=0,
        name='lp-detection-model',
        val=True
    )


#note that you want graphics card in that device='cpu'
#also you want hyper parameter tuning you want for best results