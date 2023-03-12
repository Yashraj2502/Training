import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def display(im_path):

    # dpi = 80
    dpi = mpl.rcParams['figure.dpi']   # Apparently the default dpi changed to 100, so to be safe in the future you can directly access the dpi from the rcParams
    im_data = plt.imread(im_path)
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()
    
def display_without_depth(im_path):

    # dpi = 80
    dpi = mpl.rcParams['figure.dpi']   # Apparently the default dpi changed to 100, so to be safe in the future you can directly access the dpi from the rcParams
    im_data = plt.imread(im_path)
    height, width = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()



# 01: INVERTED IMAGE

def invert_img(img):
    inverted_img = cv2.bitwise_not(img)
    cv2.imwrite('../data/temp/inverted.jpeg', inverted_img)
    display('../data/temp/inverted.jpeg')

# display('../data/temp/inverted.jpeg')



# 02: RESCALING



# 03: BINARIZATION

# This will automatically convert our image file into grayscale
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def binarization(img):
    gray_image = grayscale(img)
    cv2.imwrite("../data/temp/gray.jpeg", gray_image)
    display_without_depth("../data/temp/gray.jpeg")
    
    
    # Threshold
    thresh, im_bw = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # Edit 127 & 255 as per the requirement

    var = cv2.imwrite('../data/temp/bw_image.jpeg', im_bw)
    display_without_depth('../data/temp/bw_image.jpeg')
    return (var)



# 04: NOISE REMOVAL

def noise_removal(image):
    kernal = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernal, iterations=1)
    kernal = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernal, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernal)
    image = cv2.medianBlur(image, 3)
    return (image)

def remove_noise(img):
    no_noise = noise_removal(im_bw)
    var = cv2.imwrite('../data/temp/no_noise.jpeg', no_noise)
    display_without_depth('../data/temp/no_noise.jpeg')
    return (var)



# 05: DILATION & EROSION

def thin_font(image):
    image = cv2.bitwise_not(image)
    kernal = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernal, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def erosion(img):
    no_noise = '../data/temp/no_noise.jpeg'
    no_noise = cv2.imread(no_noise)
    eroded_image = thin_font(no_noise)
    var = cv2.imwrite('../data/temp/eroded_image.jpeg', eroded_image)
    display_without_depth('../data/temp/eroded_image.jpeg')
    # display('../data/temp/eroded_image.jpeg')
    return (var)

def thick_font(image):
    image = cv2.bitwise_not(image)
    kernal = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernal, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def dilation(img):
    dilated_image = thick_font(no_noise)
    var = cv2.imwrite('../data/temp/dilated_image.jpg', dilated_image)
    display_without_depth("../data/temp/dilated_image.jpeg")
    return (var)



# 06: ROTATION/DESKEWING

# Calculate skew angle of an image
def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(newImage, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print(len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("../data/temp/boxes.jpg", newImage)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

def deskew_rotation(img):
    fixed = deskew(new)
    var = cv2.imwrite('../data/temp/rotated_fixed.jpg', fixed)
    display('../data/temp/rotated_fixed.jpg')
    return (var)