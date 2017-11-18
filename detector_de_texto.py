import cv2
import numpy as np

def text_detection(img):
    '''
    Codigo copiado del modulo de text de opencv_contrib-3.3.0: "textdetection.py"

    '''

    # Extract channels to be processed individually
    channels = cv2.text.computeNMChannels(img)
    # Append negative channels to detect ER- (bright regions over dark background)
    cn = len(channels) - 1
    for c in range(0, cn):
        channels.append((255 - channels[c]))


    # Apply the default cascade classifier to each independent channel (could be done in parallel)
    print("Extracting Class Specific Extremal Regions from " + str(len(channels)) + " channels ...")
    print("    (...) this may take a while (...)")
    ROI=np.array([])
    for channel in channels:

        erc1 = cv2.text.loadClassifierNM1('trained_classifierNM1.xml')

        er1 = cv2.text.createERFilterNM1(erc1, 16, 0.00015, 0.13, 0.2, True, 0.1)

        erc2 = cv2.text.loadClassifierNM2('trained_classifierNM2.xml')
        er2 = cv2.text.createERFilterNM2(erc2, 0.5)

        regions = cv2.text.detectRegions(channel, er1, er2)

        rects=cv2.text.erGrouping(img, channel, [r.tolist() for r in regions])
        if len(rects)==1:
            ROI=np.append(ROI,rects)

    ROI=np.reshape(ROI,(-1,4))
    ROI=ROI.astype(int)
    return ROI