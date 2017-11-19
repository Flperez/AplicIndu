import numpy as np
import cv2
import math as mt
import matplotlib.pyplot as plt


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


        for i in range(0,umbralizada_corregida.shape[0]):
            cambio = 0
            print(i)
            for j in range(0,umbralizada_corregida.shape[1]-1):
                print(cambio)
                if umbralizada_corregida[umbralizada_corregida.shape[0]-i-1,j]!=umbralizada_corregida[i,j+1]:
                    cambio+=1
            if cambio == 2:
                cv2.line(self.img_color,(1,i),(100,i))


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