import cv2
from yolo_predictions import YOLO_Pred
from model_merger import ModelMerger

# Initialize the two YOLO models
yolo_model_1 = YOLO_Pred('yolov8_sea_animals_trained_50e/weights/best.onnx', 'data.yaml')
yolo_model_2 = YOLO_Pred('yolov8_trash_trained/weights/best.onnx', 'data2.yaml')

model_merger = ModelMerger(yolo_model_1, yolo_model_2)

# Open video capture
# cap = cv2.VideoCapture('Test Folder/istockphoto-1253258941-640_adpp_is.mp4')
# cap = cv2.VideoCapture('Test Folder/istockphoto-1312921346-640_adpp_is.mp4')
# cap = cv2.VideoCapture('Test Folder/istockphoto-1475351906-640_adpp_is.mp4')
# cap = cv2.VideoCapture('Under the sea_ Ocean animal moves [FREE RESOURCE].mp4')

######## To remove some starting part of video ######
# Calculate the frame index for 5 seconds
# fps = cap.get(cv2.CAP_PROP_FPS)
# skip_frames = int(10 * fps)

# Set the frame position to skip_frames
# cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)

# For LIVE WEB CAM
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if ret == False:
        print('Unable to read video')
        break

    # Merge model predictions
    merged_image = model_merger.merge_models(frame)

    cv2.imshow('Merged YOLO Predictions', merged_image)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
