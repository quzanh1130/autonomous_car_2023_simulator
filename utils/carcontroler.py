import numpy
import cv2
import time
from lane.lane_line_detection import laneDetector
from utils.PID_Fuzzy import Fuzzy_PID

class CarController:
    def __init__(self, param):
        self.img = 0
        self.param = param
        self.detectlane = laneDetector()
        self.turning_time = 0
        self.seeSignTime = 0
        self.last_sign_time = 0
        self.lastSignDetection = ''
        self.lastSign = ''
        self.countObject = 0
        self.haveObject = 0
        self.throttle = param.maxThrotle
        self.steering_angle = param.steering
        
        self.pid_controller = Fuzzy_PID(9,0,0.8,0,1.2, 0)
        self.pid_controller.setKp(9, 0)
        self.pid_controller .setKi(0.8, 0)
        self.pid_controller.setKd(1.2, 0)
        self.pid_controller.setSampleTime(0.015) # Set the sample time (adjust as needed)
        setpoint = 0.0
        self.pid_controller.setSetPoint(setpoint)
        
        
    def calculate_control_signal(self, objects, draw = None):
        
        # Find left/right points  
        self.left_point, self.right_point, self.haveLeft, self.haveRight, self.haveLeft2, self.haveRight2, self.len_line = self.detectlane.find_left_right_points(self.img, draw = draw)

        # Calculate deviation from the center of the lane
        im_center = self.img.shape[1] // 2

        objects_value = objects[:]
        if objects_value!=[] and objects_value[0][0] < self.im_width/2 and objects_value[0][1] < self.im_height *0.8:
            self.countObject += 1 
            self.haveObject = -1    
        elif objects_value!=[] and objects_value[0][0] > self.im_width/2 and objects_value[0][1] < self.im_height *0.8: 
            self.countObject += 1 
            self.haveObject = 1
        elif objects_value ==[] and self.left_point < 270 and self.right_point > 370:
            self.countObject = 0
            self.haveObject = 0          

        if self.left_point != -1 and self.right_point != -1:
                    
            middle_point = (self.right_point + self.left_point) // 2
            
            mid_right_point = (middle_point + self.right_point) // 2
            x_offset = im_center - mid_right_point
            
            if self.countObject >= self.param.maxCountObject and self.haveObject == -1:
                print("Object on the left road")
                 
            elif self.countObject >= self.param.maxCountObject and self.haveObject == 1:
                mid_left_point = (middle_point + self.left_point) // 2
                x_offset = im_center - mid_left_point            
                print("Object on the right road")

            self.pid_controller.update(x_offset)

            # Get the calculated steering angle from the PID controller
            steering_angle_fuzzy = self.pid_controller.output

            # Normalize the steering angle to the range -1 to 1
            steering_angle_normalized = -float(steering_angle_fuzzy/90)
            # print("x-offset: ", x_offset)

            # Clip the steering angle to the range -1 to 1
            steering_angle_normalized = max(-1, min(1, steering_angle_normalized))

            if steering_angle_normalized < -0.65 or steering_angle_normalized > 0.65 or self.countObject >= 2:
                self.throttle = 0.25
            else:
                self.throttle = 0.35

        else:
            steering_angle_normalized = 0

        self.steering_angle = steering_angle_normalized
        
        
        
        
    def decision_control(self, image, signs, objects, draw = None):
        self.img = image
        self.im_height, self.im_width = self.img.shape[:2]
        self.calculate_control_signal(objects, draw = draw)
        
        signs_value = signs[:]
        if signs_value != [] and self.lastSignDetection == '':
            for sign in signs:
                class_name, x, y, w, h = sign
                if class_name == 'left':
                    self.lastSignDetection = 'left'
                    self.seeSignTime = time.time()
                elif class_name == 'right':
                    self.lastSignDetection = 'right'
                    self.seeSignTime = time.time()
                # elif class_name == 'no_left':
                #     self.lastSignDetection = 'no_left'
                #     self.seeSignTime = time.time()
                # elif class_name == 'no_right':
                    # self.lastSignDetection = 'no_right'
                    # self.seeSignTime = time.time()
                # elif class_name == 'straight':
                    # self.lastSignDetection = 'straight'
                    # self.seeSignTime = time.time()
                elif class_name == 'stop' and self.lastSign != 'stop':
                    self.lastSignDetection = 'stop'
                    self.seeSignTime = time.time()
        
        print("lastSignDetection: " +str(self.lastSignDetection))

        # if see the sign will decreace the throttle
        if self.steering_angle != 0 and self.lastSignDetection != '' and self.turning_time == 0:  
            self.throttle = self.param.minThrottle 


           
        # will go ahead when see the sign when it don't see one or two line
        if self.lastSignDetection != '' and self.turning_time == 0 and (self.haveLeft == 0 or self.haveRight ==0) and signs_value ==[]:
            self.steering_angle = self.param.steering 
        
        # set turning time for right
        if self.steering_angle == 0 and self.turning_time == 0 and self.lastSignDetection == 'right' and signs_value ==[] and (self.haveRight ==0):
            self.turning_time = self.param.maxTurnTime
            self.last_sign_time = time.time()
            print("turn right")
        
        # set turning time for left
        if self.steering_angle == 0 and self.turning_time == 0 and self.lastSignDetection == 'left' and signs_value ==[] and (self.haveLeft == 0) :
            self.turning_time = self.param.maxTurnTime
            self.last_sign_time = time.time()
            print("turn left") 
       
         # set turning time for no_right
        if self.steering_angle == 0 and self.turning_time == 0 and self.lastSignDetection == 'no_right' and signs_value ==[]and (self.haveLeft == 0):
            self.turning_time = self.param.maxTurnTime
            self.last_sign_time = time.time()
            print("turn no_right")
        
        # set turning time for no_left
        if self.steering_angle == 0 and self.turning_time == 0 and self.lastSignDetection == 'no_left' and signs_value ==[] and (self.haveRight ==0):
            self.turning_time = self.param.maxTurnTime
            self.last_sign_time = time.time()
            print("turn no_left")
       
        # set turning time for straight
        if self.steering_angle == 0 and self.turning_time == 0 and self.lastSignDetection == 'straight' and signs_value ==[]:
            self.turning_time = self.param.maxTurnTime
            self.last_sign_time = time.time()
            print("go straight")

        # set turning time for stop sign imediately
        if self.lastSignDetection == 'stop' and self.turning_time == 0:
            self.turning_time = self.param.stoptime
            self.last_sign_time = time.time()
            print("stop")

        # if have turnning time
        if (time.time() - self.last_sign_time) >= 0 and (time.time() - self.last_sign_time) <= self.turning_time and self.lastSignDetection != '':         
            if self.lastSignDetection != '':
                if self.lastSignDetection == 'left':
                    self.throttle = 0.1
                    self.steering_angle = -1
                elif self.lastSignDetection == 'right':
                    self.throttle = 0.1
                    self.steering_angle = 1
                # elif self.lastSignDetection == 'no_left':
                #     self.throttle = 0.1
                #     self.steering_angle = abs(steering_angle)
                # elif self.lastSignDetection == 'no_right':
                #     self.throttle = 0.1
                    # self.steering_angle = -abs(steering_angle)
                elif self.lastSignDetection == 'straight':
                    self.throttle = 0.1
                    self.steering_angle = 0
                elif self.lastSignDetection == 'stop':
                    self.throttle = 0
                    self.steering_angle = 0
                    
            if (self.len_line == 2) and (time.time() - self.last_sign_time) >= self.param.minTurnTime:
                self.turning_time = 0
                self.lastSign = self.lastSignDetection
                self.lastSignDetection = ''
                self.seeSignTime = 0
                self.last_sign_time = 0
                self.countObject = 0
                self.haveObject = 0
                print("clear all")
        elif ((time.time() - self.last_sign_time) < 0 or (time.time() - self.last_sign_time) >= self.turning_time) and self.lastSignDetection != '' and self.turning_time != 0 and self.len_line == 2 :
            self.turning_time = 0
            self.lastSign = self.lastSignDetection
            self.lastSignDetection = ''
            self.seeSignTime = 0
            self.last_sign_time = 0
            self.countObject = 0
            self.haveObject = 0
            print("clear all")
        
        if self.haveLeft == 0 and self.haveRight == 0  and self.lastSignDetection == '':
            self.steering_angle = 1
            self.throttle = 0.1
        
        # Reset when see sign but don't turn
        if self.lastSignDetection != '' and self.turning_time == 0 and (time.time() - self.seeSignTime) >= 4:
            self.turning_time = 0
            self.lastSign = self.lastSignDetection
            self.lastSignDetection = ''
            self.last_sign_time = 0
            self.seeSignTime = 0
            self.countObject = 0
            self.haveObject = 0
            print("New round")    
            
        return self.throttle, self.steering_angle
    
    