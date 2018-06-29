# -*- coding: utf-8 -*-

# https://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html
# https://stackoverflow.com/questions/42203898/python-opencv-blob-detection-or-circle-detection
# https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/
# https://es.mathworks.com/help/images/texture-segmentation-using-gabor-filters.html
# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import glob
import natsort
import os

def build_filters():
    filters = []
    ksize = 51
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), sigma=4.0, theta=theta, lambd=10.0, gamma=1, psi=0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def metricas(gt_np,detectedlist):

    Correctos=0
    FalsosPositivos=0
    FalsosNegativos=0
    #Para cada defetcto real, se buscan los defectos detectados, si no se han detectado defecto, falsos negativos
    for i in range(0, gt_np.shape[1]):
        correcto_local = 0
        for t in range(0,detectedlist.__len__()):
            if ((detectedlist[t][0] + detectedlist[t][2]) < ((gt_np[0][i] - 15) + (gt_np[2][i] + 30)) and
                    (detectedlist[t][0]) > ((gt_np[0][i] - 15)) and
                    (detectedlist[t][1]) > ((gt_np[1][i] - 15)) and
                    (detectedlist[t][1] + detectedlist[t][3]) < (gt_np[1][i] - 15 + gt_np[3][i] + 30)):
                correcto_local = 1
        if(correcto_local!=0):
            Correctos+=1
        else:
            FalsosNegativos+=1

    #Para cada defecto detectado, se busca si corresponde a algun defecto real, de no ser así, se trata de un falso positivo
    for t in range(0, detectedlist.__len__()):
        correcto_local = 0
        for i in range(0,gt_np.shape[1]):
            if ((detectedlist[t][0] + detectedlist[t][2]) < ((gt_np[0][i] - 15) + (gt_np[2][i] + 30)) and
                    (detectedlist[t][0]) > ((gt_np[0][i] - 15)) and
                    (detectedlist[t][1]) > ((gt_np[1][i] - 15)) and
                    (detectedlist[t][1] + detectedlist[t][3]) < (gt_np[1][i] - 15 + gt_np[3][i] + 30)):
                correcto_local = 1
        if(correcto_local==0):
            FalsosPositivos+=1

    return Correctos, FalsosPositivos,FalsosNegativos


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required=True, help="Path folder with img and .reg to be checked")
    args = vars(ap.parse_args())
    pathin = args["folder"]

    # Almacenar las direcciones de cada imagen en una variable
    imagenes = glob.glob(pathin + '/*png')

    # Tamaño de las imágenes
    sim_imagenes = len(imagenes)

    # Ordenar las imágenes
    imagenes = natsort.natsorted(imagenes)


    for t in range(sim_imagenes):

        #Se leen las imagenes, se pasan a gris y se hacen copias para pintar encima defectos y groundtrhuth

        img = cv2.imread(imagenes[t])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgfinal = np.copy(img)
        img_ground = np.copy(img)

        #Se aplica el filtro de Gabor
        filters = build_filters()
        res1 = process(img, filters)
        (thresh, im_bw) = cv2.threshold(res1, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        #Operaciones de erosion y dilatación para obtener zonas de defectos adecuadas
        kernel_filter = np.ones((5, 5), np.uint8)
        kernel_fix = np.ones((3,3),np.uint8)
        erosion = cv2.erode(im_bw, kernel_filter, iterations=2)
        dilation = cv2.dilate(erosion, kernel_filter, iterations=2)
        ero2 = cv2.erode(dilation, kernel_fix, iterations=1)


        #cv2.imshow('result', res1)
        #cv2.waitKey(0)

        # Busqueda de contornos y dibujarlos en la imagen aquellos que correspondan a nodos
        im2, contours, hierarchy = cv2.findContours(ero2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        points=list()

        for z in range(len(contours)):
            con = contours[z]
            x, y, w, h = cv2.boundingRect(con)

            if (12 < w < 100 and 12 < h < 80) or (20 < w < 400 and 4 < h < 50):
                point = [x, y, w, h]
                cv2.rectangle(imgfinal, (x, y), (x + w, y + h), (255, 255, 255), 2)
                points.append(point)





        registro = os.path.join(pathin,str(t+1)+".reg")
        #Se carga el archivo de resitro si es posible y si no, todos los defectos encotnrados, son falsos positivos
        if os.path.isfile(registro):
            fs = cv2.FileStorage(registro, cv2.FILE_STORAGE_READ)
            fn = fs.getNode("rectangles")
            aux = fn.mat()
            #Se pinta el groundtruth
            for i in range(0, aux.shape[1]):
                cv2.rectangle(img_ground, (aux[0][i], aux[1][i]), (aux[0][i] + aux[2][i], aux[1][i] + aux[3][i]),
                              (255, 255, 255), 2)
            # Se calcula la métrica
            Correctos, FalsosPositivos, FalsosNegativos = metricas(aux, points)
        else:
            Correctos = 0
            FalsosNegativos=0
            FalsosPositivos=points.__len__()


        # Mostrar la imagen resultante
        nombre_imagen = str(t+1)
        plt.imshow(imgfinal, cmap=None), plt.title('title '+nombre_imagen), plt.show()
        plt.imshow(img_ground, cmap=None), plt.title('title '+nombre_imagen), plt.show()


        #Plots de las erosiones y dilataciones
        #plt.imshow(im_bw, cmap=None), plt.title('title'), plt.show()
        #plt.imshow(erosion, cmap=None), plt.title('title'), plt.show()
        #plt.imshow(dilation, cmap=None), plt.title('title'), plt.show()
        #plt.imshow(ero2, cmap=None), plt.title('title'), plt.show()


        #Valores de zonas de detecciones correctas, falsos positivos y falsos negativos
        print("_____________"," Imagen: ",nombre_imagen," _________________")
        print("correctos",Correctos)
        print("Falsos positivos",FalsosPositivos)
        print("Falsos negativos",FalsosNegativos)
