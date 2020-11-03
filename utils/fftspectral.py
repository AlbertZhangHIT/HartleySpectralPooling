# python this script in Python3 

from scipy import fftpack
import numpy as np
import pylab as py
import math
import datetime
 
def CropSpectrum(img, H, W):
	N = np.size(img,1)
	M = np.size(img,0)

	left = math.ceil((N - W)/2)
	top = math.ceil((M - H)/2)
	right = math.ceil((N + W)/2)
	bottom = math.ceil((M + H)/2)

	cImg = img[top:bottom, left:right]
	return cImg

def PadSpectrum(y,M,N):
	image = np.zeros((M,N),dtype=complex)
	
	H = np.size(y,0)
	W = np.size(y,1)
	
	left = math.ceil((N - W)/2)
	top = math.ceil((M - H)/2)
	right = math.ceil((N + W)/2)
	bottom = math.ceil((M + H)/2)
	
	image[top:top+H, left:left+W] = y
	return image

def TreatCornerCases(y):
	M =  np.size(y,0)
	N =  np.size(y,1)
	
	z = y
	s = []
	# mod 2, where s is the index map of z
	s.append([(0,0)])

	if M%2  == 0:
		s.append([(M // 2,0)])
	if N%2 == 0:
		s.append([(0,N // 2)])
	if M%2 == 0 and N%2 == 0:
		s.append([(M // 2,N // 2)])

	for i in s:
		z[i[0][0]][i[0][1]] = complex(z[i[0][0]][i[0][1]].real,0)

	return {'s': s, 'z': z}

def RemoveRedundency(y):
	M =  np.size(y,0)
	N =  np.size(y,1)
	result = TreatCornerCases(y)
	S = result['s']
	z = result['z']
	I = []
	for m in range(M-1):
		for n in range(int(math.floor(N/2))):
			if not ((m,n) in S) :
				if not((m,n) in I) :
					z[m][n] = 2 * z[m][n]
					I.append([(m,n),((M - m)%M, (N - n)%N)])
				else:
					z[m][n] = 0
	
	return z

def RecoverMap(y):
	M =  np.size(y,0)
	N =  np.size(y,1)
	result = TreatCornerCases(y)
	S = result['s']
	z = result['z']
	I = []
	for m in range(M-1):
		for n in range(int(math.floor(N/2))):
			if not ((m,n) in S) :
				if not((m,n) in I) :
					z[m][n] = 1/2 * z[m][n]
					z[(M-m)%M][(N-n)%N] = z[m][n]
					I.append([(m,n),((M-m)%M, (N-n)%N)])
				else:
					z[m][n] = 0
	
	return z
	

def SpectralPooling(image, H, W):
	M = np.size(image,0)
	N = np.size(image,1)

	# Take the fourier transform of the image.
	F1 = fftpack.fft2(image)

	# Now shift the quadrants around so that low spatial frequencies are in
	# the center of the 2D fourier transformed image.
	F2 = fftpack.fftshift( F1 )

	F2_crop = CropSpectrum(F2,H,W)

	result = TreatCornerCases(F2_crop)
	F2_treated = result['z']

	# Take the inverse fourier transform.
	iF2 = fftpack.ifftshift( F2_treated )
	result = np.abs(fftpack.ifft2(iF2))

	# Normalizing output to (0-1)
	amax = np.max(result)
	amin = np.min(result)
	factor = (1.0/(amax-amin))
	result = factor*(result-amin)

	return result

   
def SpectralPoolingBackward(grad_out,M,N):

	# Take the fourier transform of the grad_out.
	F1 = fftpack.fft2(grad_out)

	# Now shift the quadrants around so that low spatial frequencies are in
	# the center of the 2D fourier transformed image.
	F2 = fftpack.fftshift( F1 )

	F2_rr = RemoveRedundency(F2)

	F2_pad = PadSpectrum(F2_rr,M,N) 

	F2_rm = RecoverMap(F2_pad)

	# Take the inverse fourier transform.
	iF2 = fftpack.ifftshift( F2_rm )
	result = np.abs(fftpack.ifft2(iF2))
	
	# Normalizing output to (0-1)
	amax = np.max(result)
	amin = np.min(result)
	factor = (1.0/(amax-amin))
	result = factor*(result-amin)
	
	return result



def resize(image,H,W):
	M =  np.size(image[:,:,0],0)
	N =  np.size(image[:,:,0],1)
	
	result = np.zeros((H,W,3))
	
	# Calculating spectral pooling for each channel separately
	result[:,:,0] = SpectralPooling(image[:,:,0],H,W)
	result[:,:,1] = SpectralPooling(image[:,:,1],H,W)
	result[:,:,2] = SpectralPooling(image[:,:,2],H,W)
	return result
	
def resize_back(image,H,W):
	result = np.zeros((H,W,3))
	result[:,:,0] = SpectralPoolingBackward(image[:,:,0],H,W)
	result[:,:,1] = SpectralPoolingBackward(image[:,:,1],H,W)
	result[:,:,2] = SpectralPoolingBackward(image[:,:,2],H,W)
	return result
	
image = py.imread("img.png")
Osize = 512
Csize = 64
# forward phase

forward_time_start = datetime.datetime.now()
result = resize(image,Csize,Csize)
forward_time_end = datetime.datetime.now()
print ("Forward time:",forward_time_end-forward_time_start)
print("size: ", np.size(result, 0), np.size(result, 1))

# backward phase
backward_time_start = datetime.datetime.now()
back = resize_back(result,Osize,Osize)
backward_time_end = datetime.datetime.now()
print ("Backward time:",backward_time_end-backward_time_start)
print("size: ", np.size(back, 0), np.size(back, 1))
# Now plot up
py.figure(1)
py.clf()
py.imshow( image)
 
py.figure(2)
py.clf()
py.imshow( result)
py.savefig('FTresult.png')
 
py.figure(3)
py.clf()
py.imshow( back)
py.savefig('FTback.png')

py.xlabel('Spatial Frequency')
py.ylabel('Power Spectrum')
 
py.show()
