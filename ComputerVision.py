

import cv2 
import numpy as np 
import pytesseract

showSteps =False


def sort_contours(cnts):
	# initialize the reverse flag and sort index
	reverse = False
	i = 1
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return cnts

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


background=cv2.imread("image.jpg")

hsv = cv2.cvtColor(background,cv2.COLOR_BGR2HSV)

#By blurring the text, color and text were made to appear whole.
blur = cv2.GaussianBlur(hsv, (15,15), 0)

#Range for white
lower = np.array([0,0,0], dtype="uint8")
upper = np.array([100,100,255], dtype="uint8")

#Make the colors between these two values white and make the rest black.
mask = cv2.inRange(blur, lower, upper)
if showSteps:
    cv2.imshow("mask",cv2.resize(mask, (800,600)))
    cv2.waitKey()

#MORPH_OPEN->removing noise
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
#MORPH_DILATE->image or size of foreground object increases
mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((9,9),np.uint8))
#Inversion of input array elements.
mask = cv2.bitwise_not(mask)
if showSteps:
    cv2.imshow("maskk",cv2.resize(mask, (800,600)))
    cv2.waitKey()

#The intersections of the background and the white parts resulting from the mask are taken.
res1 = cv2.bitwise_and(background,background,mask=mask)
if showSteps:
    cv2.imshow("res",cv2.resize(res1, (800,600)))
    cv2.waitKey()



pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\Tugce\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

# Convert the image to gray scale
gray = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY) 

# Threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY) 
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9)) 
# Appplying dilation on the threshold image 
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 

# Finding contours 
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 


file = open("Text.txt", "w+")
file.write("") 
file.close() 



toBeDiscarded = []

for i, c in enumerate(contours):
    x, y, w, h = cv2.boundingRect(c) 

    if h<100 or w<100:
        toBeDiscarded.append(c)
        continue

    

for cnt in toBeDiscarded:
    contours.remove(cnt)


contours = sort_contours(contours)


for i, c in enumerate(contours):

    c = scale_contour(c, 1.1)
    # Create mask where white is what we want, black otherwise
    mask = np.zeros_like(background) 
    
    # Draw filled contour in mask
    cv2.drawContours(mask, contours, i, (255,255,255), -1) 
    if showSteps:
        cv2.imshow("1", cv2.resize(mask, (900, 400)))
        cv2.waitKey()

    out = np.ones_like(background) * 255
    if showSteps:
        cv2.imshow("2", cv2.resize(out, (900, 400)))
        cv2.waitKey()
    
    
    out[mask == 255] = background[mask == 255]
    if showSteps:
        cv2.imshow("3", cv2.resize(out, (900, 400)))
        cv2.waitKey()

    
    x, y, w, h = cv2.boundingRect(c) 
    out = out[y:y + h, x:x + w]
    if showSteps:
        cv2.imshow("out", cv2.resize(out,(900,450)))
        cv2.waitKey()
    #cv2.imshow("cropped", cv2.resize(out, (900, 400)))


    out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    th = cv2.threshold(out, 100, 255, cv2.THRESH_BINARY)[1]
    
    if showSteps:
        cv2.imshow("th",cv2.resize(th,(900,450)))
        cv2.waitKey()
    
    
    file = open("Text.txt","a") 
    text = pytesseract.image_to_string(th) 
    file.write(text) 
    file.write("\n") 
    file.close 


cv2.waitKey(0)
cv2.destroyAllWindows()