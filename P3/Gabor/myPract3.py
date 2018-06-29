import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import glob
from natsort import natsorted
import os
from functions import *


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path folder with img and .reg")
    ap.add_argument("--output", required=False, help="Path folder with result")
    ap.add_argument("--vis", action="store_true", help="Visualize results")
    args = ap.parse_args()

    lst_images = natsorted(glob.glob(args.input+"/*.png"))
    MyfilteGabor = GaborFilter()
    TotalFalsePositive,TotalTruePositive = 0,0
    for path_image in lst_images:

        # Load image and convert to gray
        img = cv2.imread(path_image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        # Apply Filter Gabors
        proccess_image = process(gray, MyfilteGabor)
        (thresh, im_bw) = cv2.threshold(proccess_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Apply Morphology functions
        kernel_filter = np.ones((5, 5), np.uint8)
        kernel_fix = np.ones((3,3),np.uint8)
        erosion = cv2.erode(im_bw, kernel_filter, iterations=2)
        dilation = cv2.dilate(erosion, kernel_filter, iterations=2)
        ero2 = cv2.erode(dilation, kernel_fix, iterations=1)


        # Get the bounding boxes
        Detected_BB = getBoxes(mask=ero2)

        # Load the ground truth
        reg_path = path_image.split('.')[0] + ".reg"
        if os.path.isfile(reg_path):
            GroundTruth_BB = load_reg(reg_path)
        else:
            GroundTruth_BB = np.array([])


        # Calculate the iou
        TruePositive, FalsePositive = verifDetection(GT_BBs=GroundTruth_BB,DT_BBs=Detected_BB,threshold=0.6)
        TotalFalsePositive+=FalsePositive
        TotalTruePositive+=TruePositive
        print(path_image,TruePositive,FalsePositive)

        # If you type "--vis"
        if args.vis:
            # Drawing the bounding boxes
            out = drawBoundingBoxes(img,GroundTruth_BB,(0,0,255))
            out = drawBoundingBoxes(out,Detected_BB,(0,255,0),thickness=1)
            cv2.imshow("result",out)
            cv2.waitKey()

    print("\n\n-----------------TOTAL--------------------")
    print("Falses positives: ",TotalFalsePositive)
    print("Trues positives: ",TotalTruePositive)



