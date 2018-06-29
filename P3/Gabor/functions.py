import numpy as np
import cv2


def GaborFilter(ksize=51,sigma=4.0,lambd=10.0,gamma=1,psi=0):
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

def load_reg(path):
    fs = cv2.FileStorage(path,cv2.FILE_STORAGE_READ)
    rect = fs.getNode("rectangles")
    return rect.mat().reshape(-1,4)



def getBoxes(mask):
    _, contours, _= cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (12 < w < 100 and 12 < h < 80) or (20 < w < 400 and 4 < h < 50):
            box = np.array([x,y,x+w,y+h])
            boxes.append(box)
    return boxes

def drawBoundingBoxes(image,boxes,color=(255,0,0)):
    if len(image.shape)==2:
        out = cv2.cvtColor(image,cv2.cvtColor(cv2.COLOR_GRAY2BGR))
    else:
        out = image.copy()
    for box in boxes:
        cv2.rectangle(out, (box[0], box[1]), ( box[2], box[3]), color, 2)
    return out

def calculateIoU(BB1,BB2):

    return IoU


def verifDetection(GT_BBs,DT_BBs,threshold):

    return TruePositive, FalsePositive
