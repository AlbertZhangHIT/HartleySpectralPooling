import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


# parameter setting
DataDir = "mnist/resnet/"

resultDir1 = DataDir + "regular16"
resultDir2 = DataDir + "hartley15"
resultDir3 = DataDir + "cosine15"
label1 = 'ResNet16'
label2 = 'DHTSP-ResNet15'
label3 = 'DCTSP-ResNet15'
TrainLogName = "TrainLog"
ValLogName = "ValLog"


num = 1 		# experiment number
epochs = 10
TrainIterEpoch = 600
ValIterEpoch = 100

error_phase = True


# Plot beggins
TrainIter = epochs*TrainIterEpoch
x_axis = np.arange(1, TrainIter+1, 1)


ValIter = epochs*ValIterEpoch
sub_axis = np.arange(1, ValIter+1, 1)

xmajorLocatorT = MultipleLocator(TrainIterEpoch)
xmajorLocatorV = MultipleLocator(ValIterEpoch)
x_epoch = [0]
x_epoch = np.append(x_epoch, np.arange(0, epochs+1).tolist())

#  Training accuracy
fig1, ax1 = plt.subplots()
#  Validation accuracy
fig2, ax2 = plt.subplots()
#  Training loss
fig3, ax3 = plt.subplots()
#  Validation loss
fig4, ax4 = plt.subplots()

acc_label = 'Accuracy'
if error_phase:
	acc_label = 'Error'
for i in range(1, num+1):
	# specify log file 
	TrainLog1 = os.path.join(resultDir1, TrainLogName+str(i)+".out")
	ValLog1 = os.path.join(resultDir1, ValLogName+str(i)+".out")

	TrainLog2 = os.path.join(resultDir2, TrainLogName+str(i)+".out")
	ValLog2 = os.path.join(resultDir2, ValLogName+str(i)+".out")

	TrainLog3 = os.path.join(resultDir3, TrainLogName+str(i)+".out")
	ValLog3 = os.path.join(resultDir3, ValLogName+str(i)+".out")

	# load log file
	train_logger1 = np.loadtxt(TrainLog1)
	val_logger1 = np.loadtxt(ValLog1)
	train_logger2 = np.loadtxt(TrainLog2)
	val_logger2 = np.loadtxt(ValLog2)
	train_logger3 = np.loadtxt(TrainLog3)
	val_logger3 = np.loadtxt(ValLog3)
	#print(train_logger.shape)
	# resize log array, 
	# [batch_time.val, batch_time.avg, losses.val, losses.avg, top1.val, top1.avg]
	train_logger1.resize((TrainIter, 6))
	val_logger1.resize((ValIter, 6))
	train_logger2.resize((TrainIter, 6))
	val_logger2.resize((ValIter, 6))
	train_logger3.resize((TrainIter, 6))
	val_logger3.resize((ValIter, 6))
	# plot
	if error_phase:
		train_logger1[:, -1] = 100. - train_logger1[:, -1]
		train_logger2[:, -1] = 100. - train_logger2[:, -1]
		train_logger3[:, -1] = 100. - train_logger3[:, -1]
		val_logger1[:, -1] = 100. - val_logger1[:, -1]
		val_logger2[:, -1] = 100. - val_logger2[:, -1]
		val_logger3[:, -1] = 100. - val_logger3[:, -1]
	# Train Acc
	l11, = ax1.plot(x_axis, train_logger1[:,-1], color='green', linestyle='dashed',
				label=label1)
	l12, = ax1.plot(x_axis, train_logger2[:,-1], color='red', linestyle='dashed',
				label=label2)
	l13, = ax1.plot(x_axis, train_logger3[:,-1], color='blue', linestyle='dashed',
			label=label3)
	ax1in = zoomed_inset_axes(ax1, 2.5, loc=5)
	l11, = ax1in.plot(x_axis, train_logger1[:,-1], color='green', linestyle='dashed',
				label=label1)
	l12, = ax1in.plot(x_axis, train_logger2[:,-1], color='red', linestyle='dashed',
				label=label2)
	l13, = ax1in.plot(x_axis, train_logger3[:,-1], color='blue', linestyle='dashed',
				label=label3)	
	ax1in.set_xlim(10,1000)
	ax1in.set_ylim(0,20)
	ax1in.get_xaxis().set_visible(False)
	ax1in.get_yaxis().set_visible(False)
	# Val Acc
	l21, = ax2.plot(sub_axis, val_logger1[:,-1], color='green', linestyle='dashed',
				label=label1)
	l22, = ax2.plot(sub_axis, val_logger2[:,-1], color='red', linestyle='dashed',
				label=label2)
	l23, = ax2.plot(sub_axis, val_logger3[:,-1], color='blue', linestyle='dashed',
				label=label3)	
	# Train Loss
	l31, = ax3.plot(x_axis, train_logger1[:,-3], color='green', linestyle='dashed',
				label=label1)
	l32, = ax3.plot(x_axis, train_logger2[:,-3], color='red', linestyle='dashed',
				label=label2)
	l33, = ax3.plot(x_axis, train_logger3[:,-3], color='blue', linestyle='dashed',
				label=label3)	
	ax3in = zoomed_inset_axes(ax3, 2.5, loc=5)
	l31, = ax3in.plot(x_axis, train_logger1[:,-3], color='green', linestyle='dashed',
				label=label1)
	l32, = ax3in.plot(x_axis, train_logger2[:,-3], color='red', linestyle='dashed',
				label=label2)
	l33, = ax3in.plot(x_axis, train_logger3[:,-3], color='blue', linestyle='dashed',
				label=label3)	
	ax3in.set_xlim(100,1000)
	ax3in.set_ylim(0,0.5)
	ax3in.get_xaxis().set_visible(False)
	ax3in.get_yaxis().set_visible(False)
	# Val Loss
	l41, = ax4.plot(sub_axis, val_logger1[:,-3], color='green', linestyle='dashed',
				label=label1)
	l42, = ax4.plot(sub_axis, val_logger2[:,-3], color='red', linestyle='dashed',
				label=label2)
	l43, = ax4.plot(sub_axis, val_logger2[:,-3], color='blue', linestyle='dashed',
				label=label3)
#### Train Phase

# accuracy
ax1.set_title('Training '+acc_label)
ax1.legend(handles=[l11, l12, l13],
			labels=[label1, label2, label3],
			loc='best')
#ax1.xaxis.set_major_locator(xmajorLocatorT)
#ax1.set_xticklabels(x_epoch)
ax1.grid(True)
#ax1.set_xlabel('epochs')
ax1.set_xlabel('iters')
ax1.set_ylabel(acc_label+'(%)')
mark_inset(ax1, ax1in, loc1=2, loc2=4, fc="none", ec="0.5")
# loss
ax3.set_title('Training Loss')
ax3.legend(handles=[l31, l32, l33],
			labels=[label1, label2, label3],
			loc='best')
#ax3.xaxis.set_major_locator(xmajorLocatorT)
#ax3.set_xticklabels(x_epoch)
ax3.grid(True)
#ax3.set_xlabel('epochs')
ax3.set_xlabel('iters')
ax3.set_ylabel('Loss')
mark_inset(ax3, ax3in, loc1=2, loc2=4, fc="none", ec="0.5")
#### Test Phase

# accuracy
ax2.set_title('Testing '+acc_label)
ax2.legend(handles=[l21, l22, l23],
			labels=[label1, label2, label3],
			loc='best')
ax2.xaxis.set_major_locator(xmajorLocatorV)
ax2.set_xticklabels(x_epoch)
ax2.grid(True)
ax2.set_xlabel('epochs')
#ax2.set_xlabel('iters')
ax2.set_ylabel(acc_label+'(%)')

# loss
ax4.set_title('Testing Loss')
ax4.legend(handles=[l41, l42, l43],
			labels=[label1, label2, label3],
			loc='best')
ax4.xaxis.set_major_locator(xmajorLocatorV)
ax4.set_xticklabels(x_epoch)
ax4.grid(True)
ax4.set_xlabel('epochs')
#ax4.set_xlabel('iters')
ax4.set_ylabel('loss')

# save figures
fig1.savefig(DataDir+'compare_Train_Acc.png', bbox_inches='tight')
fig2.savefig(DataDir+'compare_Val_Acc.png', bbox_inches='tight')
fig3.savefig(DataDir+'compare_Train_Loss.png', bbox_inches='tight')
fig4.savefig(DataDir+'compare_Val_Loss.png', bbox_inches='tight')

fig1.savefig(DataDir+'compare_Train_Acc.eps', format='eps', dpi=300)
fig2.savefig(DataDir+'compare_Val_Acc.eps', format='eps', dpi=300)
fig3.savefig(DataDir+'compare_Train_Loss.eps', format='eps', dpi=300)
fig4.savefig(DataDir+'compare_Val_Loss.eps', format='eps', dpi=300)