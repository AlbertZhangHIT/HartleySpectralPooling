import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os

DataDir = "mnist/simple/"
maxPoolDir = DataDir + 'max/'
hartleyPoolDir = DataDir + 'hartley/'
cosinePoolDir = DataDir + 'cosine/'

resultDir1 = maxPoolDir
resultDir2 = hartleyPoolDir 
resultDir3 = cosinePoolDir
label1 = 'Max'
label2 = 'Hartley Spectral'
label3 = 'Cosine Spectral'

num = 10
epochs = 10
batchEpoch = 100
iterEpoch = 600
error_phase = True
# Plot begins
x_axis = np.arange(1, epochs*iterEpoch+1, 1)
sub_axis = np.arange(1, epochs+1, 1)

Acc1 = np.zeros([num, epochs])
Acc2 = np.zeros([num, epochs])
Acc3 = np.zeros([num, epochs])

AccMean1 = np.zeros(epochs)
AccStd1 = np.zeros(epochs)
AccMean2 = np.zeros(epochs)
AccStd2 = np.zeros(epochs)
AccMean3 = np.zeros(epochs)
AccStd3 = np.zeros(epochs)

acc_label = ' Accuracy'
if error_phase:
	acc_label = 'Error'

for i in range(num):
	Fname = 'ValLog' + str(i+1) +'.out'
	logFile1 = resultDir1 + Fname
	logFile2 = resultDir2 + Fname
	logFile3 = resultDir3 + Fname

	log1 = np.loadtxt(logFile1)
	log2 = np.loadtxt(logFile2)
	log3 = np.loadtxt(logFile3)

	Acc1[i,:] = log1[::6*batchEpoch]
	Acc2[i,:] = log2[::6*batchEpoch]
	Acc3[i,:] = log3[::6*batchEpoch]

for i in range(epochs):
	AccMean1[i] = np.mean(Acc1[:,i])
	AccStd1[i] = np.std(Acc1[:,i])
	AccMean2[i] = np.mean(Acc2[:,i])
	AccStd2[i] = np.std(Acc2[:,i])
	AccMean3[i] = np.mean(Acc3[:,i])
	AccStd3[i] = np.std(Acc3[:,i])

if error_phase:
	AccMean1 = 100. - AccMean1
	AccMean2 = 100. - AccMean2
	AccMean3 = 100. - AccMean3
	

fig, ax = plt.subplots()
ax.errorbar(sub_axis, AccMean1, yerr=AccStd1,
            linestyle="dashed", color='y', ecolor='green', marker='o',
            visible=True, capsize=3,
            label=label1)
ax.errorbar(sub_axis, AccMean2, yerr=AccStd2,
    			linestyle="dashed", color=None, ecolor='red', marker='*',
            visible=True, capsize=3,
    			label=label2)
ax.errorbar(sub_axis, AccMean3, yerr=AccStd3,
    			linestyle="dashed", color=None, ecolor='blue', marker='+',
            visible=True, capsize=3,
    			label=label3)


ax.legend(numpoints=1)

plt.grid()
plt.xlabel('epoch')
plt.ylabel(acc_label+'(%)')
plt.savefig(DataDir+"compare_AccBar_epochs_"+str(epochs)+".png")
            #, transparent=True, 
            #bbox_inches='tight', pad_inches=0)
plt.show()
