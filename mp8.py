import numpy as np
import mnist
import scipy.misc
import skimage
import csv
from matplotlib import pyplot as plt


images = mnist.train_images()
images = images[:20,:,:]
#print(images[0,:,:])

# binarize
images[np.where(images < 128 )] = -1
images[np.where(images > 0 )] = 1

noiseCoordinates = np.zeros( (20, 15, 2) )
i = 0;

# read in noise coordinates
with open('./SupplementaryAndSampleData/NoiseCoordinates.csv', 'rb') as noiseCoordinateFile:
	next(noiseCoordinateFile)
	coordReader = csv.reader(noiseCoordinateFile)
	for row in coordReader:
		noiseCoordinates[i/2,:,i%2] = row[1:]
		i = i + 1

noiseCoordinates = noiseCoordinates.astype(int)
noiseImages = np.copy(images)

# noise images
for i in np.arange(20):
	#print(noiseCoordinates[i,:,1])
	noiseImages[i,noiseCoordinates[i,:,0],noiseCoordinates[i,:,1]] = -1 * images[i,noiseCoordinates[i,:,0],noiseCoordinates[i,:,1]]

# check for correctness
#print(np.count_nonzero(np.not_equal(images, noiseImages)))

#plt.imshow(noiseImages[0,:,:], interpolation = 'nearest')
#plt.show()

# Read in Initial Parameters (same for every image)
Q_init = np.zeros( (1, 28, 28) )
i = 0
with open('./SupplementaryAndSampleData/InitialParametersModel.csv', 'rb') as initParameterFile:
	paramReader = csv.reader(initParameterFile)
	for row in paramReader:
		Q_init[0, i,:] = row
		i = i + 1

# Read in order matrix (different for every image)
pixelOrder = np.zeros( (20, 28 * 28, 2) ) # N_image x N_pixel x dim
i = 0
with open('./SupplementaryAndSampleData/UpdateOrderCoordinates.csv', 'rb') as updateOrderFile:
	next(updateOrderFile)
	orderReader = csv.reader(updateOrderFile)
	for row in orderReader:
		pixelOrder[i/2,:,i%2] = row[1:]
		i = i+1
pixelOrder = pixelOrder.astype(int)

# Define constants
thetaHH = 0.8
thetaHX = 2
eps = 10**(-10)

# Initial energy terms
Eq_init = np.zeros( (1, 20) )
H = np.ones( (1, 28, 28) )
H[Q_init < .5] = -1


# Start MFI and calculate energy terms
#Q_new = np.ones( (20, 28, 28) ) * Q_init
Q = np.ones( (20, 28, 28) ) * Q_init
#test_denom = np.zeros(28)
for iter in np.arange(10):
	#Q_old = np.copy(Q_new)
	for im in np.arange(20):
		for pxl in np.arange(28*28):
			r = pixelOrder[im,pxl,0] # update row pixel
			c = pixelOrder[im,pxl,1] # update col pixel
			X = noiseImages[im,r,c]
			hSum = 0
			if r - 1 > -1:
				hSum += thetaHH * (2 * Q[im, r-1, c] - 1)
			if r + 1 < 28:
				hSum += thetaHH * (2 * Q[im, r+1, c] - 1)
			if c - 1 > -1:
				hSum += thetaHH * (2 * Q[im, r, c-1] - 1)
			if c + 1 < 28:
				hSum += thetaHH * (2 * Q[im, r, c+1] - 1)
			xSum = thetaHX * X
			logqiPos = hSum + xSum
			Q[im, r, c]  = np.exp(logqiPos) / (np.exp(logqiPos) + np.exp(-logqiPos))
			#print((np.exp(logqiPos) + np.exp(-logqiPos)))



#print(Q[1,:,:])
cleanImages = np.ones( (20, 28, 28) )
cleanImages[Q < 0.5] = 0

plt.imshow(cleanImages[1,:,:], interpolation='nearest')
#plt.imshow(noiseImages[0,:,:], interpolation='nearest')
plt.show()