import numpy as np
import cv2
import math as mt
import matplotlib.pyplot as plt
from filtro import filtro
import operator
import os,re

class caja():
    def __init__(self,x,y,w,h):
        self.xmin = x
        self.ymin = y
        self.w = w
        self.h = h
        self.xmax = x+w
        self.ymax = y+h
        self.area = w*h

def find_limits(list_cajas):
    fin = 0

    #find ymin
    while fin == 0:
        ymin = min(list_cajas, key=operator.attrgetter('ymin')).ymin
        salida = 0
        for i in range(0,len(list_cajas)):
           # print("ymin: ",ymin,"list_caja.ymin: ",list_cajas[i].ymin)
            if abs(ymin-list_cajas[i].ymin)>50:
                list_cajas = [item for item in list_cajas if item.ymin != ymin]
                salida = 1
                break
        if salida == 0:
            fin = 1

    fin = 0

    # find ymax
    ymax = max(list_cajas, key=operator.attrgetter('ymax')).ymax

    return ymin,ymax


class crotal():
    offset = 5
    def __init__(self,path,method="vecino"):

        # Nos interesa tener la imagen a color para luego pintar la recta reconocida
        self.img_color = cv2.imread(path)
        if self.img_color is None:
            print(path," imagen vacia")
            self.text=""
        else:



            # Utilizaremos siempre la imagen en escalada de grises
            img_gray = cv2.cvtColor(src=self.img_color,code=cv2.COLOR_RGB2GRAY)
            img_gray = img_gray[0:img_gray.shape[0] - self.offset, :]

            # Calculamos su imagen umbralizada
            self.img_umbralizada = crotal.calcula_img_umbralizada(self=self,img=img_gray,method = method)

            # Calculamos el angulo en el que se encuentra girada el crotal y la corregimos
            self.angle = crotal.calcula_angle(self=self,img_umbralizada=self.img_umbralizada,img=self.img_color)
            M = cv2.getRotationMatrix2D((self.img_umbralizada.shape[1] / 2, self.img_umbralizada.shape[0] / 2), self.angle, 1)
            self.img_corregida = cv2.warpAffine(img_gray,M,(self.img_umbralizada.shape[1],self.img_umbralizada.shape[0]))

            # Una vez corregida la imagen pasamos a reconocer el texto
            self.text_image = crotal.detection_text(self=self,img=self.img_corregida)
            self.text = crotal.tesseract(self=self,img=self.text_image,path=path)





    def calcula_img_umbralizada(self,img,method):

        self.hist = cv2.calcHist([img], [0], None, [255], [0, 255])
        if method == "vecino":
            #Bucle
            vecino = 5
            maximos =np.array([])
            for i in range(0+vecino,255-vecino):
                value = self.hist[i]
                rango = self.hist[i-vecino:i+vecino]
                rang = np.delete(rango,[vecino])
                if value >np.amax(rang):
                    maximos = np.append(maximos,i)

            self.th = ((maximos[1] + maximos[2]) / 2)
            _,img_umbralizada = cv2.threshold(img, self.th, 255, cv2.THRESH_BINARY)

        if method == "adaptative":
            img_umbralizada = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 11, 2)
            kernel = np.ones((3,3),np.uint8)
            img_umbralizada = cv2.erode(img_umbralizada,kernel,iterations=0)
            self.th = 0

        if method == "otsu":
            self.th, img_umbralizada = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return img_umbralizada







    def calcula_angle(self,img_umbralizada,img):

        # Buscamos la caja que contiene al crotal
        _, contours, hier = cv2.findContours(img_umbralizada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        j = np.argmax(areas)
        cnt = contours[j]
        rect = cv2.minAreaRect(cnt)

        # Calculamos el angulo
        box = cv2.boxPoints(rect)
        inccol=abs(box[1][0]-box[2][0])
        incfil=abs(box[1][1]-box[2][1])

        #Angulo
        angle = -mt.atan2(incfil,inccol)*(180/mt.pi)


        #Corregimos angulo
        if angle < -45:
            angle=angle+90

        return angle

    def detection_text(self,img):

        # Corregimos la imagen umbralizada
        M = cv2.getRotationMatrix2D((self.img_umbralizada.shape[1] / 2, self.img_umbralizada.shape[0] / 2), self.angle,
                                    1)
        umbralizada_corregida = cv2.warpAffine(self.img_umbralizada, M, (self.img_umbralizada.shape[1], self.img_umbralizada.shape[0]))


        # Se calcula la caja contenedora del crotal de nuevo para calcular su w y h
        _, contours, hier = cv2.findContours(umbralizada_corregida, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        j = np.argmax(areas)  # indice del contorno con mayor area
        x, y, w, h = cv2.boundingRect(contours[j])

        # Recortamos la imagen
        image_to_recognise = umbralizada_corregida[int(y+h/2):int(y+h),x+10:int(x+w)-10]
        image_to_recognise = 255 * np.ones(image_to_recognise.shape, dtype="uint8") - image_to_recognise

        # Aplicamos n dilataciones para mejorar los resultados
        kernel = np.ones((3,3),dtype="uint8")
        image_to_recognise = cv2.dilate(image_to_recognise,kernel,iterations=0)

        # Calcumalos todas las cajas contenidas en la imagen
        _, contours, hier = cv2.findContours(image_to_recognise, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ROI = np.array([])
        for i in range(0,len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            if w<h:
                ROI = np.append(ROI,[x,y,x+w,y+h])

        # Modificamos las dimensiones
        ROI = np.reshape(ROI,(-1,4))
        ROI = ROI.astype(int)


        # Filtramos las cajas solapadas o contenidas
        mifiltro = filtro(ROI)
        ROI = mifiltro.filtrar()

        # Creamos una lista de objetos (caja) para ordenadarlas de mayor a menor
        list_cajas = [caja(x=ROI[i,0],y=ROI[i,1],w=ROI[i,2]-ROI[i,0],h=ROI[i,3]-ROI[i,1]) for i in range(0,len(ROI))]
        list_cajas = list(reversed(sorted(list_cajas, key=operator.attrgetter('area'))))
        list_cajas = list_cajas[:4]


        '''
        areasROI = [(ROI[k, 0] - ROI[k, 2]) * (ROI[k, 1] - ROI[k, 3]) for k in range(0, len(ROI))]
        list_ordered = sorted(areasROI)
        list_ordered = list(reversed(list_ordered))
        list_ordered = list_ordered[:4]

        result = cv2.cvtColor(src=image_to_recognise, code=cv2.COLOR_GRAY2RGB)

        for i in range(0, len(list_ordered)):
            index = areasROI.index(list_ordered[i])
            box = ROI[index, :]
            cv2.rectangle(result, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        plt.imshow(result), plt.title("Cajas seleccionadas"), plt.show()
        '''

        if list_cajas:

            #Buscamos los limites de las cajas
            ymin,ymax = find_limits(list_cajas)
        else:
            ymin=0
            ymax=1

        # Volvemos a recortar la imagen para seleccionar nuestra Region de interes
        result = cv2.cvtColor(src=image_to_recognise, code=cv2.COLOR_GRAY2RGB)
        text_image = result[ymin:ymax,:]
        plt.imshow(text_image), plt.title("ROI to tesseract"), plt.show()

        return text_image

    def tesseract(self,img,path):

        # Ruta donde se guarda las imagenes
        path_out = "/home/f/PycharmProjects/AplicIndu/CrotalesTest/build"
        filename, file_extension = os.path.splitext(path)
        base = os.path.basename(path)
        name = os.path.splitext(base)[0]
        new_name = name+"_ROI"+file_extension
        path = os.path.join(path_out,new_name)

        # Guardamos imagen
        cv2.imwrite(path,img)

        #Ejecutamos tesseract-ocr como una linea de comando
        command = "tesseract "+path+" texto -l  eng -psm 7"
        os.system(command)

        #Eliminamos cualquier caracter no numerico
        string = open('texto.txt').read()
        new_str = re.sub("[^0-9]", "", string)
        open('texto_modif.txt', 'w').write(new_str)

        file = open("texto_modif.txt", "r")
        file = file.read()
        number = re.findall(r"[-+]?\d*\.\d+|\d+", file)

        if number:
            number = int(number[0])
        else:
            # Si no se ha detectado nada
            number = 0
        return number



