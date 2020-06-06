import cv2
import numpy as np


def nothing():
    # do any operation if you want
    pass

#Start the WebCam. 0 means default camera it will use 
cap = cv2.VideoCapture(0)

# img = cv2.imread('tiger.jpg')

# create tracker to find the appropriate color of the object you want to detect.
# In my case below point are used to detect the blue color 
cv2.namedWindow("Tracker")
cv2.createTrackbar("L-H", "Tracker", 0, 180, nothing)  # 0,97,62 # 138,203,225
cv2.createTrackbar("L-S", "Tracker", 97, 255, nothing)
cv2.createTrackbar("L-V", "Tracker", 62, 255, nothing)
cv2.createTrackbar("U-H", "Tracker", 138, 180, nothing)
cv2.createTrackbar("U-S", "Tracker", 203, 225, nothing)
cv2.createTrackbar("U-V", "Tracker", 225, 225, nothing)

# set the font 
font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

# infinite loop
while True:

    _, frame = cap.read()
    
    # Converts images from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # update the color space to identify the color which we detected. (Blue color)
    L_H = cv2.getTrackbarPos("L-H", "Tracker")
    L_S = cv2.getTrackbarPos("L-S", "Tracker")
    L_V = cv2.getTrackbarPos("L-v", "Tracker")
    U_H = cv2.getTrackbarPos("U-H", "Tracker")
    U_S = cv2.getTrackbarPos("U-S", "Tracker")
    U_V = cv2.getTrackbarPos("U-V", "Tracker")
     
    lower_red = np.array([L_H, L_S, L_V])  # [110, 50, 50] #0,97,62,L_H, L_S, L_V
    upper_red = np.array([U_H, U_S, U_V])  # [130, 255, 255] #138,203,225,U_H, U_S, U_V

    # Here we are defining range of blue color in HSV
    # This creates a mask of blue coloured
    # objects found in the frame.
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # using erod method to remove the black pixel from the image. It will check the size of the pixel.
    # If the size of pixel is greater than 4 the erase that pixel from the frame
    # create a kernal means a square pixel of black color
    kernal = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask,kernal)
    # contours detection
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        # we do this because there is some noise in the background while detection
        # so to avoid those we are checking if the area is large then only draw the contours else do noting
        if area > 400:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 3)

            if len(approx) == 4:
                cv2.putText(frame, "Rectangle", (x, y), font,  1, (0, 0, 0))
                # print("Its a Ractangle")
            elif 10 < len(approx) < 20:
                cv2.putText(frame, "Circle", (x, y), font, 1, (0, 0, 0))
            elif len(approx) == 3:
                cv2.putText(frame, "Triangle", (x, y), font, 1, (0, 0, 0))


    cv2.imshow("Frame", frame)
    cv2.imshow("MASK", mask)
    key = cv2.waitKey(1)
    if key == 27: # 27 means press ESC button to exit
        break
    # End if


cap.release()
cv2.destroyAllWindows()

