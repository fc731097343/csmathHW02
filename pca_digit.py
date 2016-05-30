import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def get_key(point,result):
    dis = []
    for item in result.T:
        d = np.linalg.norm(point - item)
        dis.append(d)
    key = dis.index(min(dis))
    return key

f = open("optdigits-orig.tra")
endline = 0
lines = f.readlines()
features = []
for line in lines:
    if len(line) < 4 and line.find("3") > 0:
        startline = endline - 32
        fea = []
        for fealine in range(startline,endline):
            for item in lines[fealine].strip():
                fea.append(int(item))
        features.append(fea)
    endline = endline + 1


TrainSet = np.array(features,dtype=np.uint8)
print TrainSet.shape
mean = np.mean(TrainSet, axis=0)
centered_data = TrainSet - mean

U, sigma, V = np.linalg.svd(centered_data.T, full_matrices=True)
result = np.dot(U[:,:2].T,centered_data.T)

print result.shape

xRange = [min(result[0]),max(result[0])]

yRange = [min(result[1]),max(result[1])]

x_pace = np.linspace(xRange[0], xRange[1], 7)[1:6]
y_pace = np.linspace(yRange[0], yRange[1], 7)[1:6]

keys = []
for i in range(5):
    for j in range(5):
        point = np.array([x_pace[i],y_pace[4-j]])
        keys.append(get_key(point,result))


xmajorLocator   = MultipleLocator(2)
xmajorFormatter = FormatStrFormatter('%1.1f')
xminorLocator   = MultipleLocator(1)

ymajorLocator   = MultipleLocator(5)
ymajorFormatter = FormatStrFormatter('%1.1f')
yminorLocator   = MultipleLocator(2.5)

plt.scatter(result[0], result[1], 70,color ='#00ff00',marker = 'o')
for item in keys:
    plt.scatter(result[0][item], result[1][item], 80, facecolors='#ffffff', edgecolors='#ff0000')

ax = subplot(111)



font = {'family' : 'serif',
        'color'  : 'black',
        'weight' : 'bold',
        'size'   : 'medium',
        }

ax.set_ylabel('Second Principal Component',fontdict=font)
ax.set_xlabel('First Principal Component',fontdict=font)

ax.xaxis.grid(True, which='major')
ax.yaxis.grid(True, which='minor')
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_major_formatter(xmajorFormatter)

ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_major_formatter(ymajorFormatter)

ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
plt.show()

#show digits
import cv2

RowOutline = 255*np.ones([1,166],dtype=np.uint8)
ColOutline = 255*np.ones([32,1],dtype=np.uint8)

image = RowOutline

for i in range(5):
    subimg = ColOutline
    for j in range(5):
        subimg = np.c_[subimg, 255*TrainSet[keys[5*i+j],:].reshape(32,32)]
        subimg = np.c_[subimg, ColOutline]
    image = np.r_[image, subimg]
    image = np.r_[image, RowOutline]


cv2.imshow("image", image)
cv2.waitKey (0)
cv2.imwrite("3.jpg",image)
