import asyncio
import base64
import functools
import json
import multiprocessing
import os
import time
from io import BytesIO
import sys
from multiprocessing import Process, Queue
import cv2
import numpy as np
import websockets
from PIL import Image

from websockets.protocol import Side

from traffsign.traffic_sign_detection import SignDetector
from object.object_detection import ObjectDetector
from utils.carcontroler import CarController
from utils.param import Param

param = Param()

def process_traffic_sign_loop(g_image_queue, signs):
    countSign = 0
    lastSign = ''
    detect = SignDetector(param.traffic_sign_model)
    while True:
        if g_image_queue.empty():
            time.sleep(0.1)
            continue
        image = g_image_queue.get()
        # Prepare visualization image
        draw = image.copy()
        # Detect traffic signs
        detected_signs = detect.detect_traffic_signs(image, draw=draw)
        if lastSign == '' or lastSign == detected_signs:
            countSign += 1
        else:
            countSign ==0
        
        # Update the shared signs list
        if countSign >= 15:
            signs[:] = detected_signs
        else:
            signs[:] = []
        # Show the result to a window
        # cv2.imshow("Traffic signs", draw)
        # cv2.waitKey(1)
  
def process_object_loop(g_image_queue, objects):
    detect = ObjectDetector(param.cascade, param.onnx_session)
    while True:
        if g_image_queue.empty():
            time.sleep(0.1)
            continue
        image = g_image_queue.get()
        
        draw = image.copy()
        
        detected_objects = detect.detect_object(image, draw= draw)

        # Update the shared signs list
        objects[:] = detected_objects
        
        """ DRAW FRAME """
        # cv2.imshow('Detection object', draw)
        # cv2.waitKey(1)

def save_image_to_dataset(g_image_queue, steering_angle_data, throttle_data):
    while True:
        if g_image_queue.empty():
            time.sleep(0.1)
            continue
        dataset_dir = "dataset"

        # Create the "dataset" folder if it doesn't exist
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        image = g_image_queue.get()
        # Create a unique name for the image based on the current time
        timestamp = int(time.time() * 1000)
        image_filename = f"{timestamp}_steer_{steering_angle_data.value}_throttle_{throttle_data.value}.png"
        image_path = os.path.join(dataset_dir, image_filename)
        # Save the image to the dataset folder
        print(f"Saved image: {image_filename}")
        cv2.imwrite(image_path, image)
        cv2.waitKey(1)
               

async def process_image(websocket, path, signs, objects):
    carcontrol = CarController(param)
    async for message in websocket:
        # Get image from simulation
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (640, 480))
        
        draw = image.copy()
        # Prepare visualization image

        # Update image to g_image_queue - used to run sign detection
        if not g_image_queue.full():
            g_image_queue.put(image)
  
        throttle, steering_angle = carcontrol.decision_control(image, signs, objects, draw = draw)
            
        # throttle = 0.1
        # steering_angle = 0

        """ DRAW FRAME """
        cv2.imshow('Result', draw)
        cv2.waitKey(1)
        
        # Send back throttle and steering angle
        message = json.dumps(
            {"throttle": throttle, "steering": steering_angle})
        print(message)
        await websocket.send(message)


async def main():
    process_image_partial = functools.partial(process_image, signs=signs, objects=objects)
    async with websockets.serve(process_image_partial, "0.0.0.0", 4567, ping_interval=None):
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    # Create a managed dictionary to share a global variable across processes
    manager = multiprocessing.Manager()
    g_image_queue = Queue(maxsize=5)
    signs = manager.list()
    objects = manager.list()
    steering_angle_data = manager.Value("d",0.0)
    throttle_data = manager.Value("d",0.0)
    p = Process(target=process_traffic_sign_loop, args=(g_image_queue, signs))
    o = Process(target=process_object_loop, args=(g_image_queue, objects))
    s = Process(target=save_image_to_dataset, args=(g_image_queue, steering_angle_data, throttle_data))
    p.start()
    o.start()
    # s.start()
    asyncio.run(main())
