import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from P1.crotal import crotal

path_to_directory = "CrotalesTest/TestSamples"

if __name__ == "__main__":


    ################ Prueba con un solo crotal ####################
    prueba = True
    if prueba == True:
        microtal = crotal(path='/home/f/PycharmProjects/AplicIndu/CrotalesTest/TestSamples/0002.TIF')
        #pintamos los resultados del algoritmo
        fig,axes = plt.subplots(nrows=2,ncols=2)
        axes[0, 0].imshow(microtal.img_color,cmap="gray")
        axes[0, 0].set_title("Imagen")
        axes[0, 1].hist(microtal.hist, 256, [0, 256])
        axes[0, 1].hist(microtal.th,100,[0,256],color = 'y')
        axes[0, 1].set_title("Histrograma")
        axes[1, 0].imshow(microtal.img_umbralizada, cmap="gray")
        axes[1, 0].set_title("Imagen umbralizada")
        axes[1, 1].imshow(microtal.img_corregida, cmap="gray")
        axes[1, 1].set_title("Imagen girada "+'%.2f'%microtal.angle+"'")
        plt.show()

        print("Se ha detectado: ",microtal.text)



    #######################   Test ################################
    test = False
    if test == True:
        reader = list(csv.reader(open("CrotalesTest/GroundTruth.csv")))


        # Crotales en el que no se ha reconocido ningun texto
        Nempty = 0
        empty = np.array([])

        # Crotales en los que el texto reconocido no coincide
        Nfail = 0
        fail = np.array([])

        # Crotales reconocido con exito
        Nwin = 0
        win = np.array([])
        i = 0

        print("----------- Iniciamos test -----------")
        list_files = sorted(glob.glob((path_to_directory + '/*TIF')))
        for infile in list_files:
            i+=1
            file, ext = os.path.splitext(infile)
            print("Para el crotal: ",infile)
            microtal = crotal(infile)

            #No se ha reconocido texto
            if not microtal.text or microtal.text==0:

                print("\tNo se ha reconocido texto")
                Nempty += 1
                empty = np.append(empty,i)


            elif microtal.text == int(reader[i][1]):
                print("\tEl algoritmo ha detectado: ", microtal.text)
                print("\tEl texto coincide con ", reader[i][1])
                Nwin += 1
                win = np.append(win,i)

            else:
                print("\tEl algoritmo ha detectado: ",microtal.text)
                print("\tEl texto NO coincide con ",reader[i][1])
                Nfail+=1
                fail = np.append(fail,i)


        print("\n----------- Resultados -----------")

        print("Numero de vacios: ",Nempty,"-> ",Nempty/i,"%")
        print("Numero de fallos: ",Nfail,"-> ",Nfail/i,"%")
        print("Numero de aciertos: ",Nwin,"-> ",Nwin/i,"%")


        # Pintamos en un grafico de barras los resultados
        N = 3
        ind = np.arange(N)
        total = Nempty+Nfail+Nwin
        data = [100/total*Nempty,100/total*Nfail,100/total*Nwin]

        fig, ax = plt.subplots()
        plt.ylim(0, 100)
        plt.bar(ind,data)
        plt.xticks(ind,('Vacio', 'Fallos', 'Aciertos'))
        plt.ylabel('Porcentaje(%)')
        plt.xlabel('Resultados')
        plt.title('0 iteraciones')
        plt.show()




