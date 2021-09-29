# coding = utf-8
import numpy as np
import matplotlib.pyplot as plt
import csv

with open('../rrest-syn_csv/dataset_res.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile)
	res = [row for row in reader]

with open('../rrest-syn_csv/ground_truth.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile)
	gt = [row for row in reader]

res = np.array(res[0])
gt = np.array(gt[0])

res = [float(x) for x in res]
gt = [float(x) for x in gt]
res = [round(x, 3) for x in res]
gt = [round(x, 3) for x in gt]

print res[0]
for x in range(0, 192):
	print x, res[x], gt[x]

plt.title("Evaluation on Synthetic Dataset")
plt.xlim((0, 192))
plt.ylim((20.0, 210.0))
plt.xlabel("Sample ID")
plt.ylabel("Heart Rate")
plt.scatter(range(0,192), gt, color='red', s = 2, label = 'Ground Truth')
plt.scatter(range(0,192), res, color='blue', s = 1,label = 'Experimental Result')
plt.grid(True)
plt.legend()
plt.show()

