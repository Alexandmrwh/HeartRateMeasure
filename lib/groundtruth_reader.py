# !/usr/bin/python
# coding = utf-8
import re
import csv

def extract(filename):
    p = re.compile('Simulated heart rate: (\d+) beats/min')
    f = open(filename)
    g = p.search(' '.join(f.readlines()))
    return g.groups(1)



ground_truth = []
for i in range(1, 10):
	gt = extract('../rrest-syn_csv/rrest-syn00%d_fix.txt'%i)
	ground_truth.append(gt[0])

for i in range(10, 100):
	gt = extract('../rrest-syn_csv/rrest-syn0%d_fix.txt'%i)
	ground_truth.append(gt[0])

for i in range(100, 193):
	gt = extract('../rrest-syn_csv/rrest-syn%d_fix.txt'%i)
	ground_truth.append(gt[0])

print ground_truth

with open('../rrest-syn_csv/ground_truth.csv', 'wb') as csvfile:
	writer = csv.writer(csvfile, delimiter = ',')
	writer.writerow(ground_truth)
