from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
# face_detector = YOLO(model_path)

# results = model.predict("/path/to/your/image", save=True) # saves the result in runs/detect/predict

class FaceDetector():
    def __init__(self, repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt"):

        self.model_path = '../models/face_detector_YOLOv8n.pt' # hf_hub_download(repo_id=repo_id, filename=filename)
        self.face_detector = YOLO(self.model_path)

    def predict(self, image_path):
        return self.face_detector.predict(image_path, save=True)


