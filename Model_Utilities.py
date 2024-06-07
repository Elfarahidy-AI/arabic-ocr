import cv2 as cv
import os
import joblib
import os
import joblib
import pywt
import numpy as np
import cv2 as cv
from scipy.fftpack import dct

def histogramAndCenterOfMass(img):
    h = img.shape[0]
    w = img.shape[1]
    histogram=[]
    sumX=0
    sumY=0
    num=0
    for x in range(0,w):
        localHist=0
        for y in range (0,h):
            if(img[y,x]==0):
                sumX+=x
                sumY+=y
                num+=1
                localHist+=1
        histogram.append(localHist)

    return sumX/num , sumY/num, histogram

def whiteBlackRatio(img):
    h = img.shape[0]
    w = img.shape[1]
    #initialized at 1 to avoid division by zero
    blackCount=1
    whiteCount=0
    for y in range(0,h):
        for x in range (0,w):
            if (img[y,x]==0):
                blackCount+=1
            else:
                whiteCount+=1
    return whiteCount/blackCount

def blackPixelsCount(img):
    blackCount=1 #initialized at 1 to avoid division by zero when we calculate the ratios
    h = img.shape[0]
    w = img.shape[1]
    for y in range(0,h):
        for x in range (0,w):
            if (img[y,x]==0):
                blackCount+=1

    return blackCount

def horizontalTransitions(img):
    h = img.shape[0]
    w = img.shape[1]
    maximum=0
    for y in range(0,h):
        prev=img[y,0]
        transitions=0
        for x in range (1,w):
            if (img[y,x]!=prev):
                transitions+=1
                prev= img[y,x]
        maximum= max(maximum,transitions)

    return maximum

def verticalTransitions(img):
    h = img.shape[0]
    w = img.shape[1]
    maximum=0
    for x in range(0,w):
        prev=img[0,x]
        transitions=0
        for y in range (1,h):
            if (img[y,x]!=prev):
                transitions+=1
                prev= img[y,x]
        maximum= max(maximum,transitions)

    return maximum

def number_of_endpoints(img):
    # Apply morphological operations to find endpoints
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    dilated_img = cv.dilate(img, kernel)
    endpoints_img = dilated_img - img

    # Count the number of endpoints
    endpoints_count = np.count_nonzero(endpoints_img)

    return endpoints_count

def number_of_loops(img):
    # Apply edge detection to the image
    edges = cv.Canny(img, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Count the number of contours with area greater than a threshold (assuming loops will have larger areas)
    loop_count = sum(1 for contour in contours if cv.contourArea(contour) > 100)  # Adjust the threshold as needed

    return loop_count

def number_of_line_crossings(img):
    # Apply Hough Line Transform to detect lines in the image
    lines = cv.HoughLines(img, 1, np.pi/180, 100)

    # Count the number of detected lines
    line_count = len(lines) if lines is not None else 0

    return line_count

def discrete_wavelet_transform(img):
    coeffs = pywt.dwt2(img, 'haar')  # 'haar' is the wavelet family, you can choose another one if needed
    LL, (LH, HL, HH) = coeffs  # LL: Approximation, LH: Horizontal detail, HL: Vertical detail, HH: Diagonal detail
    return LL.flatten(), LH.flatten(), HL.flatten(), HH.flatten()

def discrete_cosine_transform(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho').flatten()

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def removeMargins(img):
    th, threshed = cv.threshold(img, 245, 255, cv.THRESH_BINARY_INV)
    ## (2) Morph-op to remove noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))
    morphed = cv.morphologyEx(threshed, cv.MORPH_CLOSE, kernel)
    ## (3) Find the max-area contour
    cnts = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv.contourArea)[-1]
    ## (4) Crop and save it
    x,y,w,h = cv.boundingRect(cnt)
    dst = img[y:y+h, x:x+w]
    return dst

def fourier_features(img):
    img_resized = cv.resize(img, (32,32), interpolation=cv.INTER_AREA)
    f_transform = np.fft.fft2(img_resized)
    f_shift = np.fft.fftshift(f_transform)
    # Add a small constant to avoid log(0)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)  # 1e-8 is a small number to avoid log(0)
    return np.ravel(magnitude_spectrum)

def gradient_orientation_histogram(img):
    gx, gy = np.gradient(img.astype(float))
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
    histogram, _ = np.histogram(orientation, bins=9, range=(0, 180), weights=magnitude)
    return histogram

def getFeatures(img):
    x,y= img.shape
    featuresList=[]
    # first feature: height/width ratio
    featuresList.append(y/x)
    #second feature is ratio between black and white count pixels
    featuresList.append(whiteBlackRatio(img))
    #third and fourth features are the number of vertical and horizontal transitions
    featuresList.append(horizontalTransitions(img))
    featuresList.append(verticalTransitions(img))

    #print (featuresList)
    #splitting the image into 4 images
    topLeft=img[0:y//2,0:x//2]
    topRight=img[0:y//2,x//2:x]
    bottomeLeft=img[y//2:y,0:x//2]
    bottomRight=img[y//2:y,x//2:x]

    #get white to black ratio in each quarter
    featuresList.append(whiteBlackRatio(topLeft))
    featuresList.append(whiteBlackRatio(topRight))
    featuresList.append(whiteBlackRatio(bottomeLeft))
    featuresList.append(whiteBlackRatio(bottomRight))

    #the next 6 features are:
    #• Black Pixels in Region 1/ Black Pixels in Region 2.
    #• Black Pixels in Region 3/ Black Pixels in Region 4.
    #• Black Pixels in Region 1/ Black Pixels in Region 3.
    #• Black Pixels in Region 2/ Black Pixels in Region 4.
    #• Black Pixels in Region 1/ Black Pixels in Region 4
    #• Black Pixels in Region 2/ Black Pixels in Region 3.
    topLeftCount=blackPixelsCount(topLeft)
    topRightCount=blackPixelsCount(topRight)
    bottomLeftCount=blackPixelsCount(bottomeLeft)
    bottomRightCount=blackPixelsCount(bottomRight)

    featuresList.append(topLeftCount/topRightCount)
    featuresList.append(bottomLeftCount/bottomRightCount)
    featuresList.append(topLeftCount/bottomLeftCount)
    featuresList.append(topRightCount/bottomRightCount)
    featuresList.append(topLeftCount/bottomRightCount)
    featuresList.append(topRightCount/bottomLeftCount)
    #get center of mass and horizontal histogram
    xCenter, yCenter,xHistogram =histogramAndCenterOfMass(img)
    featuresList.append(xCenter)
    featuresList.append(yCenter)
    #featuresList.extend(xHistogram)
    #print(len(featuresList))


    # Structural features
    featuresList.append(number_of_loops(img))
    featuresList.append(number_of_line_crossings(img))
    featuresList.append(number_of_endpoints(img))

    # Transform-based features
    #dwt_features = discrete_wavelet_transform(img)
    #dct_features = discrete_cosine_transform(img)
    #featuresList.append(dwt_features)
    #featuresList.append(dct_features)

    featuresList.extend(gradient_orientation_histogram(img))
    featuresList.extend(fourier_features(img))

    return featuresList

#for binarization
def binary_otsus(image, filter:int=1):
    if len(image.shape) == 3:
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # Otsus Binarization
    if filter != 0:
        blur = cv.GaussianBlur(gray_img, (3,3), 0)
        binary_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    else:
        binary_img = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    # Morphological Opening
    # kernel = np.ones((3,3),np.uint8)
    # clean_img = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel)

    return binary_img


    
def labeltochar(label):
    chars = ['ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق','ك', 'ل', 'م', 'ن', 'ه', 'و','ي']
    charLabels =['ba', 'ta', 'tha', 'gim', 'ha', 'kha', 'dal' ,'thal', 'ra', 'zay', 'sin', 'shin', 'sad', 'dad', 'tah', 'za', 'ayn', 'gayn', 'fa', 'qaf', 'kaf', 'lam', 'mim', 'non', 'haa', 'waw', 'ya']
    positionsLabels=['Beginning','End','Isolated','Middle']
    if label=='alifMiddle'or label=='alifEnd':
        return 'ا'
    if label=='alifBeginning'or label=='alifIsolated':
        return 'أ'
    if label=='hamzaEnd':
        return 'ئ'
    if label=='hamzaBeginning'or label=='hamzaIsolated':
        return 'ء'
    if label=='hamzaMiddle':
        return 'ؤ'

    for i in range(len(charLabels)):
        for j in range(len(positionsLabels)):
            if label==charLabels[i]+positionsLabels[j]:
                return chars[i]

def Run(path):
    numOfFeatures = 1052
    chars = ['ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق','ك', 'ل', 'م', 'ن', 'ه', 'و','ي']
    charLabels =['ba', 'ta', 'tha', 'gim', 'ha', 'kha', 'dal' ,'thal', 'ra', 'zay', 'sin', 'shin', 'sad', 'dad', 'tah', 'za', 'ayn', 'gayn', 'fa', 'qaf', 'kaf', 'lam', 'mim', 'non', 'haa', 'waw', 'ya']
    positionsLabels=['Beginning','End','Isolated','Middle']
    
    word = ''
    classifier = joblib.load('classifier0.pkl')
    folder= getListOfFiles(path)
    count=0
    data1 = []
    for file in folder:
        img = cv.imread(file)
        img_resized = cv.resize(img, (32,32), interpolation=cv.INTER_AREA)
        gray_img = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
        cropped = removeMargins(gray_img)
        binary_img = binary_otsus(gray_img, 0)
        features = getFeatures(binary_img)
        data1.append(features)
        data2 = np.array(data1).reshape(-1, numOfFeatures)
        prediction = classifier.predict(data2)
        count=count+1
        print(f"Prediction for letter {count}: {prediction[0]}")
        #cv.imshow('image',binary_img)
        char=labeltochar(prediction[0])
        word=word+char
        data1 = []

    print(word)
    print(' ')
        #increase dataset