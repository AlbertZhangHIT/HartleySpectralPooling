import numpy as np

file = "./result/mnist/simple/cosine/bestprec.out"
run_nums = 5
acc = np.loadtxt(file)
print("Best error: %.2f, Mean error: %.2f, Std: %.2f" % 
	(100.-np.max(acc[:run_nums]), 100.-np.mean(acc[:run_nums]), np.std(acc[:run_nums])))