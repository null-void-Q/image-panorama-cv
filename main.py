import numpy as np
import time
import cv2 as cv
import matplotlib.pyplot as plt
MIN_MATCH_COUNT=1
SIFT=1
ORB=2
SURF=3
SurfHessianThreshold=400
ORBkpCount=1000;

def generatePlots(img1,img2,method=SIFT):
    (Mplots, Tplots) = testAndRecord(img1, img2, method)
    cases = ["Rotation", "Illumination", "Scale", "Noise", "Blurring"]
    i = 0
    for plot in Mplots:
        plt.plot(*zip(*plot))
        plt.suptitle('Matches vs ' + cases[i], fontsize=20)
        plt.xlabel(cases[i], fontsize=12)
        plt.ylabel('Matches', fontsize=12)
        plt.savefig('R_' + cases[i] + '_plot.png')
        plt.gcf().clear()
        i += 1
        # plt.show()

    i = 0
    for plot in Tplots:
        plt.plot(*zip(*plot))
        plt.suptitle('Time vs ' + cases[i], fontsize=20)
        plt.xlabel(cases[i], fontsize=12)
        plt.ylabel('Time', fontsize=12)
        plt.savefig('T_' + cases[i] + '_plot.png')
        plt.gcf().clear()
        i += 1


def testAndRecord(img1,img2,method=SIFT):
    rotationPoints=[]
    URpoints=[]
    i=0
    while(i< 360 ):
        transformed=rotate_image(img2,i)
        start = time.time()
        kp1, des1 = extractAndDescribe(img1, method)
        kp2, des2 = extractAndDescribe(transformed, method)
        matches = mathKeypoints(des1, des2)
        t= time.time()-start
        rotationPoints.append((i,len(matches)))
        URpoints.append((i, t))
        i+=15

    gammaPoints = []
    UIpoints = []
    gamma=0.1
    while(gamma < 3.0):
        transformed=adjust_gamma(img2,gamma)
        start=time.time()
        kp1, des1 = extractAndDescribe(img1, method)
        kp2, des2 = extractAndDescribe(transformed, method)
        matches = mathKeypoints(des1, des2)
        t=time.time()-start
        gammaPoints.append((gamma,len(matches)))
        UIpoints.append((gamma, t))
        gamma+=0.1

    scalePoints = []
    USpoints = []
    scale = 0.2
    while (scale < 1.80):
        transformed= scale_image(img2,scale)
        start=time.time()
        kp1, des1 = extractAndDescribe(img1, method)
        kp2, des2 = extractAndDescribe(transformed, method)
        matches = mathKeypoints(des1, des2)
        t=time.time()-start
        scalePoints.append((scale, len(matches)))
        USpoints.append((scale, t))
        scale+=0.2

    noisePoints = []
    UNpoints = []
    noiseVariance =500
    while (noiseVariance < 5000):
        transformed= add_g_noise(img2,noiseVariance)
        start=time.time()
        kp1, des1 = extractAndDescribe(img1, method)
        kp2, des2 = extractAndDescribe(transformed, method)
        matches = mathKeypoints(des1, des2)
        t=time.time()-start
        noisePoints.append((noiseVariance, len(matches)))
        UNpoints.append((noiseVariance, t))
        noiseVariance+=500

    smoothingPoints = []
    UBpoints = []
    kernel = 1
    while (kernel < 13):
        transformed=smooth_image(img2,kernel,kernel+5)
        start=time.time()
        kp1, des1 = extractAndDescribe(img1, method)
        kp2, des2 = extractAndDescribe(transformed, method)
        matches = mathKeypoints(des1, des2)
        t=time.time()-start
        smoothingPoints.append((kernel, len(matches)))
        UBpoints.append((kernel, t))
        kernel+=2


    return (rotationPoints, gammaPoints, scalePoints,noisePoints,smoothingPoints),(URpoints,UIpoints,USpoints,UNpoints,UBpoints)


def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv.LUT(image, table)
def rotate_image(image,angle):
    rows, cols = image.shape
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle,1)
    dst = cv.warpAffine(image, M, (cols, rows))
    return dst
def scale_image(image,factor=2):
    res = cv.resize(image, None, fx=factor, fy=factor, interpolation=cv.INTER_CUBIC)
    return res

def add_g_noise(image,var=1000):
    mean = 0
    sigma = var ** 0.5
    gauss=np.zeros(image.shape,np.uint8)
    cv.randn(gauss,mean,sigma)
    noisy=np.add(image, gauss)
    return noisy
def smooth_image(image,kernelSize,s):
    blurred = cv.GaussianBlur(image, (kernelSize,kernelSize), s, None, s)  # FILTER (image,kernel size,sigmax,DST,sigmay)
    return  blurred
def panorama(img1,img2,img3,method=SIFT,lowesRatio=0.75,thRatio=0.5):
    kp1, des1 = extractAndDescribe(img2, method)
    kp2, des2 = extractAndDescribe(img3, method)

    matches = mathKeypoints(des1, des2,lowesRatio)

    M = homography(kp1, kp2, matches,thRatio)

    if M is None:
        SystemError

    (TM, mask) = M
    result1 = cv.warpPerspective(img3, TM, (img3.shape[1] + img2.shape[1], max(img3.shape[0],img2.shape[0])), None, cv.WARP_INVERSE_MAP)
    result1[0:img2.shape[0], 0:img2.shape[1]] = img2

    kp3, des3 = extractAndDescribe(img1, method)
    kp4, des4 = extractAndDescribe(result1, method)

    matches = mathKeypoints(des3, des4, lowesRatio)

    M = homography(kp3, kp4, matches,thRatio)

    if M is None:
        SystemError

    (TM, mask) = M
    result = cv.warpPerspective(result1, TM, (result1.shape[1] + img1.shape[1], max(result1.shape[0],img1.shape[0])), None,
                                cv.WARP_INVERSE_MAP)

    result[0:img1.shape[0], 0:img1.shape[1]] = img1
    return result

def extractAndDescribe(image,method=1):
    if(method == 1):
        sift = cv.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(image, None)
        return (kp,des)

    elif(method == 2):
        orb = cv.ORB_create(ORBkpCount)
        kp, des = orb.detectAndCompute(image,None)
        return (kp,des)
    else:
        surf = cv.xfeatures2d.SURF_create(SurfHessianThreshold)
        kp, des = surf.detectAndCompute(image, None)
        return (kp,des)


def mathKeypoints(description1,description2,ratio=0.7):
    matcher = cv.BFMatcher(cv.NORM_L2SQR)
    rawMatches = matcher.knnMatch(description1, description2, 2)
    matches = []
    for m, n in rawMatches:
        if m.distance < ratio * n.distance:
            matches.append(m)
    return matches

def homography(img1KPs,img2KPs,matches,ratio=0.5):
    if len(matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([img1KPs[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([img2KPs[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        TM, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ratio)
        matchesMask=mask.ravel().tolist()
        return (TM,matchesMask)
    else:
        return None


def showImg(image,title):
    screen_res = 512, 512
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.resizeWindow(title, window_width, window_height)
    cv.imshow(title, image)

def autocrop(image, threshold=0):
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image



#-----------------------------------------------------


scene='1'

img1= cv.imread('./dataset/'+scene+'/1.jpeg',cv.IMREAD_GRAYSCALE)
img2 = cv.imread('./dataset/'+scene+'/2.jpeg',cv.IMREAD_GRAYSCALE)
img3 = cv.imread('./dataset/'+scene+'/3.jpeg',cv.IMREAD_GRAYSCALE)

method=ORB
SurfHessianThreshold=400
ORBkpCount=1000

#generatePlots(img1,img2,method)

img1=adjust_gamma(img1,5)
#img2=smooth_image(img2,7,5)
#img3=adjust_gamma(img3,.7)



(kp1,des1)=extractAndDescribe(img1,method)
s=time.time()
(kp2,des2)=extractAndDescribe(img2,method)
matches=mathKeypoints(des1,des2)
p=time.time()-s
(kp3,des3)=extractAndDescribe(img3,method)
print(len(kp1))
print(len(kp2))
print(p)
print(len(matches))

draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0),matchesMask = None,flags = 0)
kp11=cv.drawKeypoints(img1,kp1,None,(255,0,0),0)
kp12=cv.drawKeypoints(img2,kp2,None,(255,0,0),0)
kp13=cv.drawKeypoints(img3,kp3,None,(255,0,0),0)
KPS= cv.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)
KPS2= cv.drawMatches(img2,kp2,img3,kp3,matches,None,**draw_params)

showImg(kp11,"Keypoints1")
cv.imwrite("keypoints1.jpg",kp11)
showImg(kp12,"Keypoints2")
cv.imwrite("keypoints2.jpg",kp12)
showImg(kp13,"Keypoints3")
cv.imwrite("keypoints3.jpg",kp13)
showImg(KPS,"KeyPS")
cv.imwrite("matches.jpg",KPS)
showImg(KPS2,"KeyPS2")
cv.imwrite("matches2.jpg",KPS2)
cv.waitKey(0)

rk=panorama(img1,img2,img3,method)
showImg(rk,"res")
cv.imwrite("result.jpg",rk)
cv.waitKey(0)


#generatePlots(img1,img2,method)

#----------------------------------------------------
#result=panorama(img1,img2,img3,method)
#result=autocrop(result)
#cv.imwrite("result.jpg",result)
#draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0),matchesMask = mask,flags = 0)
#KPS= cv.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)
#showImg(img1,"1")
#showImg(img2,"2")
#showImg(KPS,"KPS")
#showImg(result1,"Result1")
#showImg(result,"Result")
#cv.waitKey(0)






