# **Real-Time Underwater Imaging for Sea Animal and Trash Detection using YOLOv8** 


## FLOW OF PROJECT

\
![UNDER WATER IMAGING BY GAURAV KUMAR CHAURASIYA](https://github.com/gauravkumarchaurasiya/Real-Time-Underwater-Imaging-for-Sea-Animal-and-Trash-Detection-using-YOLOv8-/blob/main/Blue%20organic%20timeline%20infographic.png)

     
###  **Introduction:**

This report outlines the creation of a system designed to identify sea animals and detect trash in aquatic surroundings. We utilized YOLOv8, a robust object detection technique, and harnessed the power of ONNX to facilitate model integration. The project's objective was to merge multiple models, analyze video frames, overlay model results, and achieve instant outcomes via a live webcam stream.

**Step 1: Data Collection**

- The project commenced with the compilation of two distinct datasets: one featuring images of diverse sea animals,

**Classes :** **\[ Surgeonfishes, Triggerfishes, Jacks, Spadefishes, WrasseFish, Snappers, Angelfishes, Damselfishes, Parrotfishes, Tunas, Groupers, Shark, Moorish Idol, Moorish Idol, Angel, Damsel, Grouperfishes, Jackfish, Parrotfish, Shark, Snapperfish, Spade, Surgeon, Trigger, Tuna, Wrassefish, fish, jellyfish, starfish, stingray ]** 

-  and the other encompassing images showcasing various types of aquatic litter. 

**Classes :  \[ Mask, can, cellphone, electronics, gbottle, glove, metal, misc, net, pbag, pbottle, plastic, rod, sunglasses, tire ]** 

These datasets served as the foundation for training our models.

- **Step 2: Training YOLOv8 Models**

Our chosen approach was YOLOv8, a renowned object detection algorithm known for its accuracy and speed. We trained two distinct YOLOv8 models: one for recognizing sea animals and another for detecting different types of trash. The training involved exposing the algorithm to labeled images, enabling it to learn the distinguishing characteristics of sea creatures and trash items.

It returns the trained models in the form of .pt (pytorch format).

- **Step 3: Exporting to ONNX Models**

Following successful training, we converted our YOLOv8 models into ONNX format from .pt format. This conversion to ONNX facilitated seamless integration with OpenCV.

- **Step 4: Developing the Model Merger Function**

A pivotal aspect of our project was the creation of a model merger function. This function played a key role in combining the results produced by the sea animal detection and trash identification models. The output from this function provided a consolidated analysis for each video frame.

- **Step 5: Utilizing OpenCV for Model Loading**

For model implementation, we leveraged OpenCV, a widely-used computer vision library. OpenCV offers user-friendly functions for loading and operationalizing machine learning models within our code.

It able to read and  process ONNX models.

- **Step 6: Processing Video Frames in Sequence**

To enable systematic analysis, we used OpenCV to process video frames sequentially. This step-by-step approach ensured that each frame could be subjected to analysis using our models.Each frame pass to models and predict the detection.

- **Step 7: Overlaying Model Outputs for Analysis**

OpenCV functions to apply both the sea animal detection and trash identification models to each frame. Subsequently, we layered the model outcomes onto the original frame, creating a visual representation of identified sea animals and litter items.

- **Step 8: Integrating Live Webcam Feed**

The culmination of our efforts was the integration of the system with a live webcam feed. This integration allowed us to perform real-time analysis on the incoming video stream, with the combined model outputs presented in real time.

**Conclusion:**

By combining YOLOv8 and ONNX technologies carefully, we built a complete system that can find sea animals and recognize litter in water. Using OpenCV along with these methods, we could process live video from a webcam and quickly show if there are sea animals or trash in the water. This helps us take care of the environment and protect marine life.
