import numpy as np
import cv2
from hmmlearn import hmm
import os
from natsort import natsorted
import argparse
import sys
import glob
from sklearn.externals import joblib
np.random.seed(42)

# Arguments
ap = argparse.ArgumentParser()
ap.add_argument("--path", required=True, help="Path to files Train")
ap.add_argument("--out", required=True, help="Path to save model and K-means center: ../model.pkl")
ap.add_argument("--k_means",default=5, required=False,type = int, help="Number of K means")
ap.add_argument("--observables_states",default=5, required=False,type = int, help="Number of observables states Markov")
ap.add_argument("--hidden_states", default=5,required=False,type = int, help="Number of hidden states Markov")
ap.add_argument("--mode", required=True, help="mode: 'right' or 'left'")
args = ap.parse_args()



if __name__=="__main__":
    if args.mode != "right" and args.mode != "left":
        print("Type 'right' or 'left'")
        sys.exit()
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    # Loading data
    if args.mode=="right":
        lst_files = natsorted(glob.glob(args.path+"/PasilloDerecha*"))
    else:
        lst_files = natsorted(glob.glob(args.path + "/PasilloIzquierda*"))

    data = np.array([])
    for file in lst_files:
        data_file = np.loadtxt(file,dtype=np.float32,delimiter=',')[:,2:]
        data = np.append(data,data_file).reshape(-1,2).astype(np.float32)




    # K-MEANS
    #https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
    # define criteria and apply kmeans()
    sequence = np.array([])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, centers = cv2.kmeans(data, args.k_means, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    Y=[]
    for idf,files in enumerate(lst_files):
        data_file = np.loadtxt(files,dtype=np.float32,delimiter=',')[:,2:]
        Y.append(data_file.shape[0])
        for point in data_file:
            minimum = 99
            for idc,center in enumerate(centers):
                distance =  np.linalg.norm(point-center)
                if distance<minimum:
                    minimum=distance
                    sec=idc
            sequence=np.append(sequence,sec).reshape(-1,1).astype(np.int64)

    # Markov
    start_probability =(1/args.k_means)*np.ones(args.k_means)
    transition_probability = 0.2*np.ones((args.k_means,args.k_means))
    emissionprob=transition_probability
    model = hmm.MultinomialHMM(n_components=args.observables_states,
                                verbose=True, n_iter=20)
    model.emissionprob = emissionprob
    model.start_probability = start_probability
    model.transition_probability = transition_probability
    model.fit(sequence, Y)

    ## Saving data
    joblib.dump(model, os.path.join(args.out,"model_"+args.mode+".pkl"))
    np.savetxt(os.path.join(args.out,"center_"+args.mode+"_"+str(args.k_means)+".txt"),centers)