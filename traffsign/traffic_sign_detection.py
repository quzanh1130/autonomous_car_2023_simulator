import cv2
import numpy as np
from utils.detection import ObjectFinder


class SignDetector:
    def __init__(self, model):
        self.object_finder = ObjectFinder(hsv_color_ranges=((0, 50),), min_saturation=127, min_value=10, min_sign_size=25, differential_rate=0.2)
        self.sign_classifier = model
        self.classes = ['unknown', 'left', 'no_left', 'right', 'no_right', 'straight', 'stop']

    def detect_traffic_signs(self, img, draw = None):

      points = self.object_finder.find_by_color(img)

      # Preprocess
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = img.astype(np.float32)
      img = img / 255.0

      # Classify signs using CNN
      signs = []
      for bbox in points:
          # Crop sign area
          x, y, w, h = bbox
          sub_image = img[y:y+h, x:x+w]

          if sub_image.shape[0] < 20 or sub_image.shape[1] < 20:
              continue

          # Preprocess
          sub_image = cv2.resize(sub_image, (32, 32))

          sub_image = np.expand_dims(sub_image, axis=0)

          # Use CNN to get prediction
          self.sign_classifier.setInput(sub_image)
          preds = self.sign_classifier.forward()
          preds = preds[0]
          cls = preds.argmax()
          score = preds[cls]

          # Remove unknown objects
          if cls == 0:
              continue

          # Remove low confidence objects
          if score < 0.5:
              continue

          signs.append([self.classes[cls], x, y, w, h])

          # Draw prediction result
          if draw is not None:
              text = self.classes[cls] + ' ' + str(round(score, 2))
              cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 255, 255), 4)
              cv2.putText(draw, text, (x, y-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

      return signs
