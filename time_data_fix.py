#############################################################
#															#
#	time_data_fix.py										#
#	排除訓練資料中的異常值，避免影響預測					#
#															#
#															#
#############################################################

# import necessary modules
import numpy as np
from sklearn.svm import SVR

prepared_data_path="prepared_data\\"			#特徵提取完成的資料路徑
routes=["A-2","A-3","B-1","B-3","C-1","C-3"]
files_start=["morning","night"]
KERNEL='linear'
C_VALUE=0.00035
DEL_MAPE=1

def parsePoint(line):
	values = [float(x) for x in line.replace(',', ' ').split(' ')]
	return values[0], values[1:]

avgKeep=[]
for route in routes:
	for file_start in files_start:
		fr = open(prepared_data_path + route+"_{}_training2.csv".format(file_start), 'r')
		fw_fixed = open(prepared_data_path + route+"_{}_training2_fixed.csv".format(file_start), 'w')
		lines = fr.readlines()
		features_train=[]
		labels_train=[]
		keeps=[]
		for i,line in enumerate(lines):
			label,feature=parsePoint(line)
			features_train.append(feature)
			labels_train.append(label)
		features_test=features_train	
		labels_test=labels_train
		
		X=np.array(features_train)
		y=np.array(labels_train)
		X_test=np.array(features_test)
		svr_rbf = SVR(kernel=KERNEL, C=C_VALUE)
		y_rbf = svr_rbf.fit(X, y).predict(X_test)
		keep=0
		for i,predict in enumerate(y_rbf):
			mape=abs(labels_test[i]-predict)/labels_test[i]
			if(mape>=DEL_MAPE):
				#異常不保留
				keeps.append(False)
			else:
				keeps.append(True)
				keep+=1
		for i,line in enumerate(lines):
			if(keeps[i]):
				fw_fixed.writelines(line)
		print("資料維持率 = {}".format(keep/len(keeps)))
		avgKeep.append(keep/len(keeps))

print("平均資料維持率 = {}".format(sum(avgKeep)/len(avgKeep)))
				