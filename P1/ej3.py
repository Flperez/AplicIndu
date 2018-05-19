import numpy as np
from hmmlearn import hmm

'''
np.random.seed(42)
model = hmm.MultinomialHMM(3,verbose=True,n_iter=20)
states = ["Sol","Nublado","Lluvia"]
observaciones = ["paraguas","si paraguas"]

#iniciamos las matrices de probabilidad del modelo

model.start_probability=np.array([0.6,0.3,0.1])
model.transtion_probability = np.array([[0.5,0.4,0.1],[0.5,0.3,0.2],[0.2,0.4,0.4]])
model.emissionprob = np.array([[0.1,0.9],[0.5,0.5],[0.8,0.2]])


train1 = [1,1,1,0]
train2 = [0,0,0]
print("train1=",",".join(map(lambda x:observaciones[x],train1)))
print("train2=",",".join(map(lambda x:observaciones[x],train2)))

X = [train1,train2]
lengths = list(map(lambda x: len(x),X))
X = np.hstack(X)
X= X.reshape(len(X),1)


model = model.fit(X,lengths)

print(model.monitor)
print("Se logro convergencia ",model.monitor_converged)

train1_reformated = list(map(lambda x: [x],train1))
train2_reformated = list(map(lambda x: [x],train2))

print("log(P(train1))",model.score(train1_reformated))
print("log(P(train2))",model.score(train2_reformated))


print()
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

x = np.random.randint(25,100,25)
y = np.random.randint(175,255,25)
z = np.hstack((x,y))
z = z.reshape((50,1))
z = np.float32(z)
plt.hist(z,256,[0,256]),plt.show()

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS

# Apply KMeans
compactness,labels,centers = cv2.kmeans(z,2,None,criteria,10,flags)


A = z[labels==0]
B = z[labels==1]

# Now plot 'A' in red, 'B' in blue, 'centers' in yellow
plt.hist(A,256,[0,256],color = 'r')
plt.hist(B,256,[0,256],color = 'b')
plt.hist(centers,32,[0,256],color = 'y')
plt.show()

