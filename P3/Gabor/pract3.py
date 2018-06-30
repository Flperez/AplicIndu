import argparse
import glob
from natsort import natsorted
import os
import matplotlib.pyplot as plt
from functions import *


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path folder with img and .reg")
    ap.add_argument("--vis", action="store_true", help="Visualize results")
    args = ap.parse_args()

    path_in = args.input
    # Check if it's a image or directory
    if  ".png" in path_in.split('/')[-1]:
        lst_images = [path_in]
    else:
        lst_images = natsorted(glob.glob(args.input+"/*.png"))
    MyfilteGabor = GaborFilter()
    TotalFalsePositive,TotalTruePositive,TotalBBGT,TotalFalseNegative = 0,0,0,0
    print("PATH TO FILE\t\t\t\tTruePositive\tFalsePositive\tFalseNegative")

    for path_image in lst_images:

        # Load image and convert to gray
        img = cv2.imread(path_image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        # Apply Filter Gabors
        proccess_image = process(gray, MyfilteGabor)
        (thresh, im_bw) = cv2.threshold(proccess_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Apply Morphology functions
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(im_bw, kernel, iterations=2)
        dilation = cv2.dilate(erosion, kernel, iterations=2)
        kernel = np.ones((3, 3), np.uint8)
        ero2 = cv2.erode(dilation, kernel, iterations=1)


        # Get the bounding boxes
        Detected_BB = getBoxes(mask=ero2)

        # Load the ground truth
        reg_path = path_image.split('.')[0] + ".reg"
        if os.path.isfile(reg_path):
            GroundTruth_BB = load_reg(reg_path)
            TotalBBGT+=GroundTruth_BB.shape[0]
        else:
            GroundTruth_BB = np.array([])


        # Calculate the iou
        TruePositive, FalsePositive, FalseNegative = verifDetection(GT_BBs=GroundTruth_BB,DT_BBs=Detected_BB,threshold=0.6)
        TotalFalsePositive+=FalsePositive
        TotalTruePositive+=TruePositive
        TotalFalseNegative+=FalseNegative
        print(path_image,TruePositive,FalsePositive,FalseNegative)

        # If you type "--vis"
        if args.vis:
            # Drawing the bounding boxes
            out = drawBoundingBoxes(img,GroundTruth_BB,(0,0,255))
            out = drawBoundingBoxes(out,Detected_BB,(0,255,0),thickness=1)
            cv2.imshow("result",out)
            cv2.waitKey()

    print("\n\n-----------------TOTAL--------------------")
    print("Total of BB GT: ",TotalBBGT)
    print("Falses positives: ",TotalFalsePositive)
    print("Trues positives: ",TotalTruePositive)
    print("False negative: ", TotalFalseNegative)
    print("Recall:",TotalTruePositive/(TotalTruePositive+TotalFalseNegative))
    print("Accuracy: ",TotalTruePositive/TotalBBGT)




