import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, cascade, onnx_session):
        # List of class names
        self.classes = ["nocar","car" ]
        self.cascade = cascade
        self.onnx_session = onnx_session
    def detect_object(self, image, draw=None):
        # Detect faces using Haar Cascade
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        objects = self.cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
        detect_objects = []
        for (x, y, w, h) in objects:
            gray_face_roi = image[y:y+h, x:x+w]  # Extract grayscale face region
    
            # Preprocess the grayscale face image
            gray_face_resized = cv2.resize(gray_face_roi, (32, 32))
            gray_face_normalized = gray_face_resized.astype('float32') / 255.0
            gray_face_normalized = np.expand_dims(gray_face_normalized, axis=0)  # Add batch dimension

            # Perform prediction
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            predictions = self.onnx_session.run([output_name], {input_name: gray_face_normalized})
            # print("pre: ", predictions)

            # Get predicted class and confidence
            predicted_class = np.argmax(predictions)
            # print("Predicted class:", predicted_class)
            confidence = predictions[0][0][predicted_class]
            # print("Confidence:", confidence)
            # print("Predicted class:", class_names[predicted_class])

            # Draw bounding box and label if confidence is more than 80%
            if self.classes[predicted_class] == "car" and confidence >= 0.93:
                mid_x = x + w / 2
                mid_y = y + h / 2
                detect_objects.append((mid_x, mid_y))
                
                cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(draw, self.classes[predicted_class]+f' {str(round(confidence, 3))}', (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
        return detect_objects
