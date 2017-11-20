import numpy as np
import cv2
import math as mt
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image

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
            self.text = crotal.recognition_text(self=self,img=self.img_corregida)




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






        #Corregimos angulo
        if angle >90:
            angle=angle-90

        return angle

    def recognition_text(self,img):

        #TODO: Crear BB para el texto
        M = cv2.getRotationMatrix2D((self.img_umbralizada.shape[1] / 2, self.img_umbralizada.shape[0] / 2), self.angle,
                                    1)
        umbralizada_corregida = cv2.warpAffine(self.img_umbralizada, M, (self.img_umbralizada.shape[1], self.img_umbralizada.shape[0]))
        plt.imshow(umbralizada_corregida,cmap="gray"),plt.show()


        _, contours, hier = cv2.findContours(umbralizada_corregida, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        result = cv2.cvtColor(src=umbralizada_corregida, code=cv2.COLOR_GRAY2RGB)
        areas = [cv2.contourArea(c) for c in contours]
        j = np.argmax(areas)  # indice del contorno con mayor area
        x, y, w, h = cv2.boundingRect(contours[j])
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.imshow(result), plt.show()

        image_to_recognise = umbralizada_corregida[int(y+h/2):int(y+h),x:int(x+w)]
        image_to_recognise = 255 * np.ones(image_to_recognise.shape, dtype="uint8") - image_to_recognise
        kernel = np.ones((3,3),dtype="uint8")

        image_to_recognise = cv2.erode(image_to_recognise,kernel,iterations=1)
        plt.imshow(image_to_recognise, cmap="gray"), plt.show()


        _, contours, hier = cv2.findContours(image_to_recognise, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        result = cv2.cvtColor(src=image_to_recognise,code=cv2.COLOR_GRAY2RGB)

        ROI = np.array([])
        for i in range(0,len(contours)):
            cv2.drawContours(image=result,contours=contours,contourIdx=i,color=(255,0,0))
            x, y, w, h = cv2.boundingRect(contours[i])
            ROI = np.append(ROI,[x,y,x+w,y+h])
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ROI = np.reshape(ROI,(-1,4))






        plt.imshow(result), plt.show()

        text = Image.fromarray(image_to_recognise)
        text = pytesseract.image_to_string(text)
        print("text: ",text)





        #TODO: Pasar tesseract
        return "0288"




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