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
    '''
    while fin == 0:
        ymax = max(list_cajas, key=operator.attrgetter('ymax')).ymax
        salida = 0
        for i in range(0, len(list_cajas)):
            if abs(ymax - list_cajas[i].ymax) > 50:
                list_cajas = [item for item in list_cajas if item.ymax != ymax]
                salida = 1
        if salida == 0:
            fin = 1
    '''
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
        '''
        edges = cv2.Canny(img_umbralizada, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)
        angle = 0
        if lines.all():
            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            angle = theta*(180/mt.pi)
        '''


        _, contours, hier = cv2.findContours(img_umbralizada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        result = cv2.cvtColor(src=img_umbralizada, code=cv2.COLOR_GRAY2RGB)
        areas = [cv2.contourArea(c) for c in contours]
        j = np.argmax(areas)  # indice del contorno con mayor area
        cnt = contours[j]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)

        inccol=abs(box[1][0]-box[2][0])
        incfil=abs(box[1][1]-box[2][1])

        angle = -mt.atan2(incfil,inccol)*(180/mt.pi)
        box = np.int0(box)
        cv2.drawContours(result, [box], 0, (0, 0, 255), 2)
        plt.imshow(result),plt.title("Calculo angulo"), plt.show()




        #Corregimos angulo
        if angle < -45:
            angle=angle+90

        return angle

    def detection_text(self,img):

        M = cv2.getRotationMatrix2D((self.img_umbralizada.shape[1] / 2, self.img_umbralizada.shape[0] / 2), self.angle,
                                    1)
        umbralizada_corregida = cv2.warpAffine(self.img_umbralizada, M, (self.img_umbralizada.shape[1], self.img_umbralizada.shape[0]))
        plt.imshow(umbralizada_corregida,cmap="gray"),plt.title("umbralizado corregido"),plt.show()


        _, contours, hier = cv2.findContours(umbralizada_corregida, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        result = cv2.cvtColor(src=umbralizada_corregida, code=cv2.COLOR_GRAY2RGB)
        areas = [cv2.contourArea(c) for c in contours]
        j = np.argmax(areas)  # indice del contorno con mayor area
        x, y, w, h = cv2.boundingRect(contours[j])
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.imshow(result),plt.title("umbralizado corregido con BB"), plt.show()

        image_to_recognise = umbralizada_corregida[int(y+h/2):int(y+h),x+10:int(x+w)-10]
        image_to_recognise = 255 * np.ones(image_to_recognise.shape, dtype="uint8") - image_to_recognise
        kernel = np.ones((3,3),dtype="uint8")

        image_to_recognise = cv2.dilate(image_to_recognise,kernel,iterations=0)
        plt.imshow(image_to_recognise, cmap="gray"),plt.title("Imagen en la que buscamos BB"), plt.show()


        _, contours, hier = cv2.findContours(image_to_recognise, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        result = cv2.cvtColor(src=image_to_recognise,code=cv2.COLOR_GRAY2RGB)

        ROI = np.array([])
        for i in range(0,len(contours)):
            cv2.drawContours(image=result,contours=contours,contourIdx=i,color=(255,0,0))
            x, y, w, h = cv2.boundingRect(contours[i])
            if w<h:
                ROI = np.append(ROI,[x,y,x+w,y+h])
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ROI = np.reshape(ROI,(-1,4))
        ROI = ROI.astype(int)
        result = cv2.cvtColor(src=image_to_recognise, code=cv2.COLOR_GRAY2RGB)
        for i in range(0, len(ROI)):
            cv2.drawContours(image=result, contours=contours, contourIdx=i, color=(255, 0, 0))
            cv2.rectangle(result, (ROI[i,0],ROI[i,1] ), (ROI[i,2], ROI[i,3]), (0, 255, 0), 2)
        plt.imshow(result),plt.title("Cajas sin filtrar"), plt.show()




        mifiltro = filtro(ROI)
        ROI = mifiltro.filtrar()
        list_cajas = [caja(x=ROI[i,0],y=ROI[i,1],w=ROI[i,2]-ROI[i,0],h=ROI[i,3]-ROI[i,1]) for i in range(0,len(ROI))]
        list_cajas = list(reversed(sorted(list_cajas, key=operator.attrgetter('area'))))
        list_cajas=list_cajas[:4]



        result = cv2.cvtColor(src=image_to_recognise, code=cv2.COLOR_GRAY2RGB)
        for i in range(0, len(ROI)):
            cv2.drawContours(image=result, contours=contours, contourIdx=i, color=(255, 0, 0))
            cv2.rectangle(result, (ROI[i, 0], ROI[i, 1]), (ROI[i, 2], ROI[i, 3]), (0, 255, 0), 2)
        plt.imshow(result), plt.title("Cajas filtradas"), plt.show()

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

        if list_cajas:
            ymin,ymax = find_limits(list_cajas)
        else:
            ymin=0
            ymax=1

        result = cv2.cvtColor(src=image_to_recognise, code=cv2.COLOR_GRAY2RGB)
        text_image = result[ymin:ymax,:]
        plt.imshow(text_image), plt.title("ROI to tesseract"), plt.show()

        return text_image

    def tesseract(self,img,path):
        path_out = "/home/f/PycharmProjects/AplicIndu/CrotalesTest/build"
        filename, file_extension = os.path.splitext(path)
        base = os.path.basename(path)
        name = os.path.splitext(base)[0]
        new_name = name+"_ROI"+file_extension
        path = path_out+'/'+new_name
        cv2.imwrite(path,img)
        command = "tesseract "+path+" texto -l  eng -psm 7"
        #print(command)
        os.system(command)

        #Eliminamos cualquier caracter no numerico
        string = open('texto.txt').read()
        new_str = re.sub("[^0-9]", "", string)
        open('texto_modif.txt', 'w').write(new_str)

        #abrimos el archivo modificado
        file = open("texto_modif.txt", "r")
        file = file.read()
        number = re.findall(r"[-+]?\d*\.\d+|\d+", file)
        if number:
            number = int(number[0])
        else:
            number = 0
        return number





'''

        #Apply k-means
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
        # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)

        # Set flags (Just to avoid line break in the code)
        flags = cv2.KMEANS_PP_CENTERS

        # Apply KMeans
        compactness, labels, centers = cv2.kmeans(hist, 3, None, criteria, 15, flags) 
'''
'''
     lines = cv2.HoughLinesP(erosion, 1, np.pi / 180, 100, 50, 10)

     if lines.all():
         for i in range(lines.shape[0]):
             [x1, y1, x2, y2] = lines[i, 0, :]
             ang = mt.atan2(y2 - y1, x2 - x1)
             cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
             angle = np.append(angle, ang)
'''
'''

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(image_to_recognise, cv2.MORPH_OPEN, kernel, iterations=2)

        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        result = cv2.cvtColor(src=image_to_recognise,code=cv2.COLOR_GRAY2RGB)

        markers = cv2.watershed(result,markers)
        result[markers == -1] = [255, 0, 0]
        
                plt.imshow(result), plt.show()

'''