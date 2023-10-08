import logging
import math
import cv2
import numpy as np

import os
import time

class laneDetector():
    def __init__(self):
        self.image = 0
        self.roi1 = 0.83
        self.roi2 = 0.9
        self.showImg = True
        

    def enhance_white_color(self, gamma=5):
        # Convert the input image to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to enhance contrast
        equ = cv2.equalizeHist(gray)

        # Convert back to color image with enhanced white color
        enhanced_img = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)

        # Apply gamma correction for contrast adjustment
        inv_gamma = 1.0 / gamma
        enhanced_img = np.power(enhanced_img / 255.0, inv_gamma)
        enhanced_img = np.uint8(enhanced_img * 255)
        return enhanced_img

    def show_image(self, title, frame):
        if self.showImg:
            cv2.imshow(title, frame)
    
    def shadow_remove(self, img):
        rgb_planes = cv2.split(img)
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_norm_planes.append(norm_img)
        shadowremove = cv2.merge(result_norm_planes)
        return shadowremove
    
    def birdview_transform(self, image):
        """
            Apply bird-view transform to the image
        """
        IMAGE_H = 480
        IMAGE_W = 640
        src = np.float32([[0, IMAGE_H], [640, IMAGE_H], [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
        dst = np.float32([[240, IMAGE_H], [640 - 240, IMAGE_H], [-160, 0], [IMAGE_W+160, 0]])
        M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
        warped_img = cv2.warpPerspective(image, M, (IMAGE_W, IMAGE_H)) # Image warping
        return warped_img

    def draw_image_with_filled_contour(self, mask):
        # Create a copy of the mask
        mask_copy = mask.copy()
        
        # Draw a white line at the bottom of the mask
        cv2.line(mask_copy, (0, mask_copy.shape[0] - 1), (mask_copy.shape[1], mask_copy.shape[0] - 1), (255, 255, 255), 2)
        # Thực hiện flood fill sau khi đã vẽ dòng trắng

        # Tìm các contour trong mask_copy
        contours, hierarchy = cv2.findContours(mask_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Chuyển mask_copy sang ảnh RGB để vẽ contour
        mask_rgb = cv2.cvtColor(mask_copy, cv2.COLOR_GRAY2RGB)
        # Tạo một ảnh trắng với cùng kích thước như mask_copy
        filled_contour = np.ones_like(mask_rgb) * 255
        
        for i, contour in enumerate(contours):
            # Lấy thông tin về phần tử cha của contour
            parent_idx = hierarchy[0][i][3]
            
            # Nếu không có phần tử cha (parent_idx == -1), vẽ contour
            if parent_idx == -1:
                
                area = cv2.contourArea(contour)
                if area >= 60000:
                    # Vẽ contour lên mask_rgb
                    # cv2.drawContours(mask_rgb, [contour], -1, (0, 255, 0), 2)
                    
                    # Vẽ filled contour bằng màu trắng
                    cv2.fillPoly(mask_rgb, [contour], (255, 255, 255))
                else:
                    # Nếu diện tích nhỏ hơn 2200, vẽ filled contour bằng màu đen
                    cv2.fillPoly(filled_contour, [contour], (0, 0, 0))
            else:
                # Vẽ filled contour bằng màu trắng cho các contour con
                cv2.fillPoly(filled_contour, [contour], (255, 255, 255))
        
        # Tạo ảnh mới bằng phép giao giữa ảnh ban đầu và filled_contour
        new_image = cv2.bitwise_and(mask_rgb, filled_contour)
        #convert new_image to binary image
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
        new_image = self.shadow_remove(new_image)
            
        return new_image

    def apply_canny_filter_and_display(self):
        
        image = cv2.GaussianBlur(self.image, (5, 5), 0)
     
        # filter for blue lane lines
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # lọc màu trắng làn đường 
        #(hMin = 2 , sMin = 8, vMin = 96), (hMax = 36 , sMax = 49, vMax = 231)
        lower1, upper1 = np.array([2, 8, 96]), np.array([36, 49, 231])
        #(hMin = 0 , sMin = 0, vMin = 45), (hMax = 127 , sMax = 63, vMax = 162)
        lower2, upper2 = np.array([0, 0, 45]), np.array([127, 63, 162])
        #(hMin = 0 , sMin = 0, vMin = 0), (hMax = 39 , sMax = 255, vMax = 104)
        lower3, upper3 = np.array([0, 0, 0]), np.array([39, 255, 104]) #lọc màu đất .
        #(hMin = 0 , sMin = 0, vMin = 0), (hMax = 45 , sMax = 28, vMax = 105)
        
        mask_1 = cv2.inRange(hsv, lower1, upper1) # dải màu đỏ thứ nhất
        mask_2 = cv2.inRange(hsv, lower2, upper2) # dải màu đỏ thứ hai
        mask_3 = cv2.inRange(hsv, lower3, upper3) # dải màu đỏ thứ ba  
        
        mask_r = cv2.bitwise_or(mask_1, mask_2) # kết hợp 2 kết quả từ 2 dải màu khác nhau
        mask_r = cv2.subtract(mask_r, mask_3) # kết hợp kết quả từ 3 dải màu khác nhau

        contours, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = 50  # Adjust this threshold as needed
        filtered_edges = np.zeros_like(mask_r)
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                cv2.drawContours(filtered_edges, [contour], 0, 255, thickness=-1)
        
        img_gauss = cv2.GaussianBlur(filtered_edges, (3, 3), 0)
        fill_image = self.draw_image_with_filled_contour(img_gauss)
        edges = cv2.Canny(fill_image, 25, 200)

        kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed
        merged_edges = cv2.dilate(edges, kernel, iterations=1)
        
        height = merged_edges.shape[0]
        two_thirds_height = (2 * height) // 3
        two_thirds_height = height - two_thirds_height
        img_thresholded = merged_edges.copy()
        img_thresholded[:two_thirds_height, :] = 0

        return img_thresholded
        

    def find_left_right_points(self, img, draw = None):
        """Find left and right points of lane
        """
        """Calculate speed and steering angle using Fuzzy PID controller"""
        self.image = img  
        self.edge = self.apply_canny_filter_and_display()
        self.img_birdview = self.birdview_transform(self.edge)
        
        draw[:, :] = self.birdview_transform(draw)
        
        im_height, im_width = self.img_birdview.shape[:2]
        # ====================================================================
        # Consider the position 70% from the top of the image
        interested_line_y = int(im_height * self.roi1)
        interested_line_y2 = int(im_height * self.roi2)
        
        interested_line_x = int(im_width * 0.25)
        interested_line_x2 = int(im_width * 0.75)
        
        if draw is not None:
            cv2.line(draw, (interested_line_x, 0),
                    (interested_line_x, im_height), (0, 0, 255), 2)
            cv2.line(draw, (interested_line_x2, 0),
                    (interested_line_x2, im_height), (0, 0, 255), 2) 
                    
            cv2.line(draw, (0, interested_line_y),
                    (im_width, interested_line_y), (0, 0, 255), 2)
            cv2.line(draw, (0, interested_line_y2),
                    (im_width, interested_line_y2), (0, 0, 255), 2) 
            
               
            
        interested_line = self.img_birdview[interested_line_y, :]
        interested_line2 = self.img_birdview[interested_line_y2, :]

        # Detect left/right points
        left_point = -1
        right_point = -1
        
        left_point2 = -1
        right_point2 = -1
        
        # lane_width = 210
        center = im_width // 2

        haveLeft = 0
        haveRight = 0
        
        haveLeft2 = 0
        haveRight2 = 0


        # Traverse the two sides, find the first non-zero value pixels, and
        # consider them as the position of the left and right lines
        for x in range(center, 0, -1):
            if interested_line[x] > 0:
                left_point = x
                haveLeft = 1
                break
        
        for x in range(center + 1, im_width):
            if interested_line[x] > 0:
                right_point = x
                haveRight = 1
                break
            
        # Traverse the two sides, find the first non-zero value pixels, and
        # consider them as the position of the left  and right lines 2
        for x in range(center, 0, -1):
            if interested_line2[x] > 0:
                left_point2 = x
                haveLeft2 = 1
                break
        
        for x in range(center + 1, im_width):
            if interested_line2[x] > 0:
                right_point2 = x
                haveRight2 = 1
                break
        
            
        if (haveLeft != 0 and haveRight !=0):
            len_line = 2
        elif (haveLeft == 0 and haveRight ==0):
            len_line = 0
        else:
            len_line = 1
        
        # ============================================================================
        # print("leftpoin 1: ", left_point)
        # print("rightpoint 1: ", right_point)
        # print("leftpoin 2: ", left_point2)
        # print("rightpoint 2: ", right_point/2)
        
        if abs(left_point - right_point2) < 30:
            right_point = left_point
            left_point = -1
        elif abs(right_point - left_point2) < 30:
            left_point = right_point 
            right_point = -1
              
        if left_point != -1 and right_point != -1:          
            if (left_point<233) and (right_point<407):
                left_point = right_point - 166
            if (left_point>233) and (right_point>407):
                right_point = left_point + 166
            if (left_point<233) and (right_point>407) and (abs(right_point-left_point)>166):
                left_point = 235
                right_point = 405

        # Predict right point when only see the left point
        if left_point != -1 and right_point == -1:
            if (left_point > 225 ) and (left_point < 245):
                right_point = left_point + 166
            else:
                right_point = left_point + 220

        # Predict left point when only see the right point
        if right_point != -1 and left_point == -1:
            if (390<right_point ) and (right_point < 410):
                left_point = right_point - 166
            else:
                left_point = right_point - 220
        # =====================================================================================

        # middle_point = (right_point + left_point) // 2
        # mid_left_point = -1
        # im_center = self.image.shape[1] // 2
        # mid_left_point = (middle_point + left_point) // 2
        if draw is not None:
            if left_point != -1:
                draw = cv2.circle(
                    draw, (left_point, interested_line_y), 7, (255, 255, 0), -1)
            if right_point != -1:
                draw = cv2.circle(
                    draw, (right_point, interested_line_y), 7, (0, 255, 0), -1)
            if left_point2 != -1:
                draw = cv2.circle(
                    draw, (left_point2, interested_line_y2), 7, (255, 255, 0), -1)
            if right_point2 != -1:
                draw = cv2.circle(
                    draw, (right_point2, interested_line_y2), 7, (0, 255, 0), -1)
                
        self.show_image("Result", draw)
        return left_point, right_point, haveLeft, haveRight, haveLeft2, haveRight2 , len_line
    


    
