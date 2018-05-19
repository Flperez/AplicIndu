import time
import numpy



print("\n\n")
time.sleep(1)
print("Ejecutando 'el toque' \n")
for i in range(0,60):
    print("\r Memoria restaurada en un ",int((i/60)*100),"%")
    time.sleep(1)
