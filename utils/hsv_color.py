import cv2
import numpy as np

def nothing(x):
    pass

def enhance_white_color(img, gamma=1.0):
    # Convert the input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to enhance contrast
    equ = cv2.equalizeHist(gray)

    # Convert back to color image with enhanced white color
    enhanced_img = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)

    # Apply gamma correction for contrast adjustment
    inv_gamma = 1.0 / gamma
    enhanced_img = np.power(enhanced_img / 255.0, inv_gamma)
    enhanced_img = np.uint8(enhanced_img * 255)

    return enhanced_img

def main():
    global output

    cv2.namedWindow("image")

    cv2.createTrackbar('HMin','image',0,179,nothing)
    cv2.createTrackbar('SMin','image',0,255,nothing)
    cv2.createTrackbar('VMin','image',0,255,nothing)
    cv2.createTrackbar('HMax','image',0,179,nothing)
    cv2.createTrackbar('SMax','image',0,255,nothing)
    cv2.createTrackbar('VMax','image',0,255,nothing)

    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    output = np.zeros((480,640,3), dtype=np.uint8)

    while True:
        cv_image = cv2.imread("/home/fptlab/Downloads/via-trafficsign-coco-20210321/train/00128.jpg")

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        hMin = cv2.getTrackbarPos('HMin','image')
        sMin = cv2.getTrackbarPos('SMin','image')
        vMin = cv2.getTrackbarPos('VMin','image')
        hMax = cv2.getTrackbarPos('HMax','image')
        sMax = cv2.getTrackbarPos('SMax','image')
        vMax = cv2.getTrackbarPos('VMax','image')

        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(cv_image, cv_image, mask=mask)

        cv2.imshow("image", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
