import numpy as np
import mnist
import scipy.misc
import skimage
import csv
from matplotlib import pyplot as plt

def calcEnergy(Q):
	Eq = np.zeros(20)
	for im in np.arange(20):
		# EX[logQ]
		Qim = np.squeeze( Q[im,:,:] )
		ElogQ = np.sum( np.sum( ( Qim * np.log( Qim + eps ) ) + ( (1-Qim) * np.log( (1-Qim) + eps ) ) ) )
		# EX[logP] (a little more involved)
		Eqk = 2 * Qim - 1 # also a 28 x 28 array
		#hSum = ( np.sum( np.sum( Eqk[0:26,:] ) * Eqk[1:27,:] ) + np.sum( np.sum( Eqk[:,0:26] * Eqk[:,1:27] ) ) ) * 2 * thetaHH
		hSum = 0
		for r in np.arange(28):
			for c in np.arange(28):
				if r - 1 > -1:
					hSum += thetaHH * Eqk[r,c] * Eqk[r-1,c]
				if r + 1 < 28:
					hSum += thetaHH * Eqk[r,c] * Eqk[r+1,c]
				if c - 1 > -1:
					hSum += thetaHH * Eqk[r,c] * Eqk[r,c-1]
				if c + 1 < 28:
					hSum += thetaHH * Eqk[r,c] * Eqk[r,c+1]
		xSum = np.sum( np.sum( Eqk * noiseImages[im,:,:] ) ) * thetaHX
		ElogP = hSum + xSum
		#if im == 1:
			#print("Normal hSum: ", hSum)
			#print(np.sum( np.sum( Eqk[ 0:26,:] ) * Eqk[1:27,:] ))
			#print(np.sum( np.sum( Eqk[:,0:26] * Eqk[:,1:27] ) ))
			#print(Eqk[0:26,:] * Eqk[1:27,:])	
			#print("Normal ElogP: ", ElogP)
		Eq[im] = ElogQ - ElogP

	return Eq

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

# Initial energy term
Eq = np.zeros( (20, 11) )
Q = np.ones( (20, 28, 28) ) * Q_init
Eq[:,0] = calcEnergy(Q)


# Start MFI and calculate energy terms
#Q_new = np.ones( (20, 28, 28) ) * Q_init
for itr in np.arange(10):
	#Q_old = np.copy(Q_new)
	for im in np.arange(20):
		for pxl in np.arange(28*28):
			r = pixelOrder[im,pxl,0] # update row pixel
			c = pixelOrder[im,pxl,1] # update col pixel
			Xj = noiseImages[im,r,c]
			hSum = 0
			if r - 1 > -1:
				hSum += thetaHH * (2 * Q[im, r-1, c] - 1)
			if r + 1 < 28:
				hSum += thetaHH * (2 * Q[im, r+1, c] - 1)
			if c - 1 > -1:
				hSum += thetaHH * (2 * Q[im, r, c-1] - 1)
			if c + 1 < 28:
				hSum += thetaHH * (2 * Q[im, r, c+1] - 1)
			xSum = thetaHX * Xj
			logqiPos = hSum + xSum
			Q[im, r, c]  = np.exp(logqiPos) / (np.exp(logqiPos) + np.exp(-logqiPos))

	Eq[:,itr+1] = calcEnergy(Q)

cleanImages = np.ones( (20, 28, 28) )
cleanImages[Q < 0.5] = 0
#for i, img in enumerate(cleanImages):
#	plt.imsave('img' + str(i) + '.jpg', img)

submitImages = np.zeros((28,280))
otherImages = np.zeros((28,280))
for i in np.arange(10):
	submitImages[:,(28*i)+np.arange(28)] = cleanImages[10+i,:,:]
	otherImages[:,(28*i)+np.arange(28)] = cleanImages[i,:,:]

np.savetxt('denoised.csv', submitImages, delimiter = ',', fmt = '%d')
np.savetxt('energy.csv', Eq[10:12,0:2], delimiter = ',', fmt = '%f')


#plt.imshow(otherImages)
#plt.show()

#plt.imshow(cleanImages[1,:,:], interpolation='nearest')
#plt.imshow(noiseImages[0,:,:], interpolation='nearest')
#plt.show()