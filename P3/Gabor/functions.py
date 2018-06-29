import numpy as np
import cv2

############# GABOR FILTER FUNCTIONS #############
def GaborFilter(ksize=51,sigma=4.0,lambd=10.0,gamma=1,psi=0):
    '''
     https://gist.github.com/odebeir/5237529
    '''
    filters = []
    ksize = ksize
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=psi, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def process(image, filters):
    accum = np.zeros_like(image)
    for kern in filters:
        fimg = cv2.filter2D(image, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

############# LOAD DATA FUNCTIONS #############

def load_reg(path):
    fs = cv2.FileStorage(path,cv2.FILE_STORAGE_READ)
    rect = fs.getNode("rectangles")
    rect = rect.mat().T.reshape(-1,4)
    rect[:,2]+= rect[:, 0]
    rect[:,3]+= rect[:, 1]

    return rect

############# GET BOUNDING BOX FUNCTION #############

def getBoxes(mask):
    _, contours, _= cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        #Limiting size of bounding box
        if (12 < w < 100 and 12 < h < 80) or (20 < w < 400 and 4 < h < 50):
            box = np.array([x,y,x+w,y+h])
            boxes.append(box)
    boxes = np.asarray(boxes,dtype=int)
    return boxes

############# DRAWING FUNCTION #############
def drawBoundingBoxes(image,boxes,color=(255,0,0),thickness=2):
    if len(image.shape)==2:
        out = cv2.cvtColor(image,cv2.cvtColor(cv2.COLOR_GRAY2BGR))
    else:
        out = image.copy()
    if boxes.size!=0:
        for box in boxes:
            cv2.rectangle(out, (box[0], box[1]), ( box[2], box[3]), color,thickness=thickness)
    return out


############# CALCULATE METRICS FUNCTIONS #############
def contained(A, B):
    corners = [[A[0],A[1]],[A[0],A[3]],[A[1],A[3]],[A[2],A[1]]]
    for corner in corners:
        if ((corner[0] >= B[0] and corner[1] >= B[1])
            and (corner[0] <= B[2] and corner[1] <= B[3])):
            return True

    if A[0]<B[0] and A[2]>B[2] and B[1]<A[1] and B[3]>A[3]:
        return True
    return False

def calculateIoU(boxA,boxB):
    '''
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    '''
    if (contained(boxA, boxB) or contained(boxB,boxA)) == False:
        iou = 0
    else:
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = (xB - xA) * (yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        iou = interArea / float(boxAArea)

    return iou


def verifDetection(GT_BBs,DT_BBs,threshold=0.6):
    TruePositive,TruePositivep,FalsePositive,FalseNegative = 0,0,0,0

    if GT_BBs.size == 0:
        FalsePositive = len(DT_BBs)
    else:
        for GT_BB in GT_BBs:
            flag_TP=False
            cntIOU=0
            for DT_BB in DT_BBs:
                iou =calculateIoU(DT_BB,GT_BB)

                if iou>threshold:
                    flag_TP=True
                    cntIOU+=1
                    TruePositive += 1
                    TruePositivep+=1

            if cntIOU>1:
                TruePositive-=(cntIOU-1)

            if not flag_TP:
                FalseNegative+=1
        if GT_BBs.shape[0]<DT_BBs.shape[0]:
            FalsePositive+=(DT_BBs.shape[0]-TruePositivep)
    return TruePositive, FalsePositive,FalseNegative
