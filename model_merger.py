import cv2
from yolo_predictions import YOLO_Pred

class ModelMerger:
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def merge_models(self, frame):
        pred_frame_1 = self.model1.predictions(frame.copy())
        pred_frame_2 = self.model2.predictions(frame.copy())
        
        merged_frame = pred_frame_1.copy()
        merged_frame = self.model2.predictions(merged_frame)
        
        return merged_frame
