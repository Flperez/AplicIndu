import numpy as np
import sklearn
import glob
from natsort import  natsorted
import argparse
from sklearn.externals import joblib


# Arguments
ap = argparse.ArgumentParser()
ap.add_argument("--path", required=True, help="Path to files or single file")
ap.add_argument("--path_modelR", required=True, help="Path to model right")
ap.add_argument("--path_modelL", required=True, help="Path to model left")
ap.add_argument("--path_centerR", required=True, help="Path to center_right.txt")
ap.add_argument("--path_centerL", required=True, help="Path to center_left.txt")

args = ap.parse_args()

if __name__=="__main__":

    # List of files
    if ".txt" in args.path.split('/')[-1]:
        lst_files = [args.path]
    else:
        lst_files = natsorted(glob.glob(args.path+"/"+"*.txt"))

    # Loading models and centers
    centerR = np.loadtxt(args.path_centerR)
    centerL = np.loadtxt(args.path_centerL)
    modelR = joblib.load(args.path_modelR)
    modelL = joblib.load(args.path_modelL)

    # Labels GT and predicted
    y_true = []
    y_pred =[]

    # Bucle of list of files
    for file in lst_files:
        # load data
        data_file = np.loadtxt(file, dtype=np.float32, delimiter=',')[:, 2:]
        # Label
        if "Paseante" in file.split('/')[-1]:
            y_true.append(0)
        elif "Derecha"  in file.split('/')[-1]:
            y_true.append(1)
        else:
            y_true.append(2)



        sequenceR = np.array([])
        sequenceL = np.array([])

        # Convert points to label center
        for point in data_file:
            minimumR = 99
            minimumL = 99

            # Right
            for idc,center in enumerate(centerR):
                distance = np.linalg.norm(point - center)
                if distance < minimumR:
                    minimumR = distance
                    sec = idc
            sequenceR = np.append(sequenceR, sec).reshape(-1, 1).astype(np.int64)

            # Left
            for idc, center in enumerate(centerL):
                distance = np.linalg.norm(point - center)
                if distance < minimumL:
                    minimumL = distance
                    sec = idc
            sequenceL = np.append(sequenceL, sec).reshape(-1, 1).astype(np.int64)

        # Inference
        scoreR = abs(modelR.score(sequenceR))
        scoreL = abs(modelL.score(sequenceL))

        if scoreL >= 100.0 and scoreR >= 100.0:
            print(file," Paseante")
            y_pred.append(0)

        else:
            if scoreR >= scoreL:
                y_pred.append(2)
                print(file," Izquierda")

            else:
                y_pred.append(1)
                print(file," Derecha")


    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    print("\n---------------------\nAccuracy: ",accuracy)