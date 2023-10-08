import cv2
import onnxruntime
import numpy as np

class Param:
     def __init__(self):
        self.minThrottle = 0.25
        self.maxThrotle = 0.35
        self.steering = 0
        
        self.minTurnTime = 0.8
        self.maxTurnTime = 2
        self.maxCountObject = 2
        self.stoptime = 4
        # Initalize traffic sign classifier
        self.traffic_sign_model = cv2.dnn.readNetFromONNX("models/best_model_checkpoint_v2.onnx")

        # Load Haar Cascade classifier
        self.cascade = cv2.CascadeClassifier('object/car.xml')

        # Load ONNX runtime session
        self.onnx_session = onnxruntime.InferenceSession('models/classification_model_v3.onnx')