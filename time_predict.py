#############################################################
#															#
#	time_predict.py											#
#	透過先前處理好的特徵資料，訓練SVR回歸模型				#
#	並產出預測結果											#
#															#
#############################################################

# import necessary modules
import numpy as np
from sklearn.svm import SVR
import math
from datetime import datetime,timedelta

prepared_data_path="prepared_data\\"
test_path="data\\"
predict_file_name="kdd2017_time_final.csv"

files=["A-2","A-3","B-1","B-3","C-1","C-3"]

KERNEL='linear'
C_VALUE=0.00035
OUTPUT_WEIGHT=0.96
weathers={}
ID=""
TID=""

#花費時間的歷史平均比率，以每小時為單位區分三個窗格
#diffWeightMorning 權重所對應的時間窗格依序是
#8:00~8:20, 8:20~8:40, 8:40~9:00, 9:00~9:20, 9:20~9:40, 9:40~10:00
diffWeightMorning={}
diffWeightMorning["A-2"]=[0.9692205889166895,1.0197299833194329,1.0110494277638773,0.9803813639712314,1.0254620447002534,0.9941565913285151]
diffWeightMorning["A-3"]=[0.9510715063161005,1.0112883723837376,1.0376401213001623,1.0898830305360332,1.0657882948748614,0.8443286745891057]
diffWeightMorning["B-1"]=[0.9451030318074102,1.0553720819290004,0.9995248862635893,0.9951078852648163,0.9985520107022794,1.0063401040329043]
diffWeightMorning["B-3"]=[0.9488236341129925,1.0108364819054865,1.040339883981521,1.0488475351160869,1.0318189493227412,0.9193335155611723]
diffWeightMorning["C-1"]=[0.9997109784560796,0.9976696501930444,1.0026193713508758,0.9766593477844416,0.9542600315350496,1.0690806206805086]
diffWeightMorning["C-3"]=[1,1,1,1,1,1]

#diffWeightNight 權重所對應的時間窗格依序是
#17:00~17:20, 17:20~17:40, 17:40~18:00, 18:00~18:20, 18:20~18:40, 18:40~19:00
diffWeightNight={}
diffWeightNight["A-2"]=[1.0326087015626009,0.9850357440176961,0.9823555544197027,1.026998700373119,0.9639475817157866,1.0090537179110943]
diffWeightNight["A-3"]=[1.0231573080780898,1.0149545271063705,0.9618881648155397,0.9636417392127569,0.9778896122313967,1.0584686485558465]
diffWeightNight["B-1"]=[1,1,1,1,1,1]
diffWeightNight["B-3"]=[1.0038916039068098,0.9939373203969888,1.0021710756962012,0.9745795086319944,1.0601868910732282,0.9652336002947777]
diffWeightNight["C-1"]=[1,1,1,1,1,1]
diffWeightNight["C-3"]=[1,1,1,1,1,1]

#取得訓練資料中，訓練用的時間窗格(7,16)平均花費的時間
def getAvgTime(features):
	times=[]
	for feature in features:
		times.append(feature[-1])
	return sum(times)/len(times)

#取得測試資料中，訓練用的時間窗格(7,16)花費的時間
def getTime(feature):
	return feature[-1]	


#載入測試資料那幾天的降雨資料
def loadWeather(in_file):
	out_suffix = ''
	in_file_name = in_file + '.csv'
	out_file_name = in_file.split('_')[1] + out_suffix + '.csv'

	fr = open(test_path + in_file_name, 'r')
	fr.readline()  # skip the header
	vol_data = fr.readlines()
	fr.close()

	for i in range(len(vol_data)):
		each_pass = vol_data[i].replace('"', '').split(',')
		precipitation = float(each_pass[8])
		hour=int(each_pass[1])
		pass_time = each_pass[0]
		pass_time = datetime.strptime(pass_time, "%Y-%m-%d")
		time_window_minute = int(math.floor(pass_time.minute / 20) * 20)
		time = datetime(pass_time.year, pass_time.month, pass_time.day,
									 pass_time.hour, time_window_minute, 0)
		if time.day not in weathers:
			weathers[time.day] = {}
		if(hour>0):
			for h in range(3):
				weathers[time.day][hour-h-1]=precipitation

def parsePoint(line):
	values = [float(x) for x in line.replace(',', ' ').split(' ')]
	return values[0], values[1:]

#針對不同時間窗格中，花費時間的歷史平均狀態作加權處理
#對於某些route的某些時間窗格(如C-3早 C-1, C-3晚)，因歷史資料數目不足，因此該route的時間窗格權重調整為1
def getDiffWindowPredict(predict,mn_time,index):
	route=ID+"-"+TID
	weight=1
	if(mn_time==1):
		weight=diffWeightMorning[route][index]
	if(mn_time==2):
		weight=diffWeightMorning[route][index]
	if(mn_time==3):
		weight=diffWeightNight[route][index]
	if(mn_time==4):
		weight=diffWeightNight[route][index]
	return predict*(math.sqrt(weight))
	
#針對降雨影響旅程時間作調整
def getWeatherPredict(predict,nowww):
	if(nowww>=2):
		return predict*math.sqrt(1.110727412879317)
	elif(nowww>=1.0):
		return predict*math.sqrt(1.0960104809326925)
	elif(nowww>0):
		return predict*math.sqrt(1.0730721851729204)
	else:
		return predict*math.sqrt(0.98)
		
#經由SVR訓練、降雨及時間窗格權重運算後，輸出預測結果
def printResult(fw,date,predict,mn_time):
	route=ID+"-"+TID
	predict=predict*OUTPUT_WEIGHT
	if(mn_time==1):
		nowww=weathers[int(date)][8]
	if(mn_time==2):
		nowww=weathers[int(date)][9]
	if(mn_time==3):
		nowww=weathers[int(date)][17]
	if(mn_time==4):
		nowww=weathers[int(date)][18]
	predict=getWeatherPredict(predict,nowww)
	if(mn_time==1):
		datestr="\"[2016-10-{} 08:00:00,2016-10-{} 08:20:00)\"".format(int(date),int(date))
		fw.writelines("{},{},{},{}".format(ID,TID,datestr,getDiffWindowPredict(predict,mn_time,0))+"\n")
		datestr="\"[2016-10-{} 08:20:00,2016-10-{} 08:40:00)\"".format(int(date),int(date))
		fw.writelines("{},{},{},{}".format(ID,TID,datestr,getDiffWindowPredict(predict,mn_time,1))+"\n")
		datestr="\"[2016-10-{} 08:40:00,2016-10-{} 09:00:00)\"".format(int(date),int(date))
		fw.writelines("{},{},{},{}".format(ID,TID,datestr,getDiffWindowPredict(predict,mn_time,2))+"\n")
	elif(mn_time==2):
		datestr="\"[2016-10-{} 09:00:00,2016-10-{} 09:20:00)\"".format(int(date),int(date))
		fw.writelines("{},{},{},{}".format(ID,TID,datestr,getDiffWindowPredict(predict,mn_time,3))+"\n")
		datestr="\"[2016-10-{} 09:20:00,2016-10-{} 09:40:00)\"".format(int(date),int(date))
		fw.writelines("{},{},{},{}".format(ID,TID,datestr,getDiffWindowPredict(predict,mn_time,4))+"\n")
		datestr="\"[2016-10-{} 09:40:00,2016-10-{} 10:00:00)\"".format(int(date),int(date))
		fw.writelines("{},{},{},{}".format(ID,TID,datestr,getDiffWindowPredict(predict,mn_time,5))+"\n")
	elif(mn_time==3):
		datestr="\"[2016-10-{} 17:00:00,2016-10-{} 17:20:00)\"".format(int(date),int(date))
		fw.writelines("{},{},{},{}".format(ID,TID,datestr,getDiffWindowPredict(predict,mn_time,0))+"\n")
		datestr="\"[2016-10-{} 17:20:00,2016-10-{} 17:40:00)\"".format(int(date),int(date))
		fw.writelines("{},{},{},{}".format(ID,TID,datestr,getDiffWindowPredict(predict,mn_time,1))+"\n")
		datestr="\"[2016-10-{} 17:40:00,2016-10-{} 18:00:00)\"".format(int(date),int(date))
		fw.writelines("{},{},{},{}".format(ID,TID,datestr,getDiffWindowPredict(predict,mn_time,2))+"\n")
	elif(mn_time==4):
		datestr="\"[2016-10-{} 18:00:00,2016-10-{} 18:20:00)\"".format(int(date),int(date))
		fw.writelines("{},{},{},{}".format(ID,TID,datestr,getDiffWindowPredict(predict,mn_time,3))+"\n")
		datestr="\"[2016-10-{} 18:20:00,2016-10-{} 18:40:00)\"".format(int(date),int(date))
		fw.writelines("{},{},{},{}".format(ID,TID,datestr,getDiffWindowPredict(predict,mn_time,4))+"\n")
		datestr="\"[2016-10-{} 18:40:00,2016-10-{} 19:00:00)\"".format(int(date),int(date))
		fw.writelines("{},{},{},{}".format(ID,TID,datestr,getDiffWindowPredict(predict,mn_time,5))+"\n")

#SVR模型訓練並預測		
def SVR_Model(fw,train_lines,test_train_lines,test_lines,mn_time):
	features_train=[]
	labels_train=[]
	features_test=[]	
	labels_test=[]
	for i,line in enumerate(train_lines):
		label,feature=parsePoint(line)
		labels_train.append(label)
		features_train.append(feature)
		
	for i,line in enumerate(test_lines):
		label,feature=parsePoint(line)
		labels_test.append(label)
		features_test.append(feature)
		
	X=np.array(features_train)
	y=np.array(labels_train)
	X_test=np.array(features_test)
	
	svr_rbf = SVR(kernel=KERNEL, C=C_VALUE)
	y_rbf = svr_rbf.fit(X, y).predict(X_test)
	
	avgTime=getAvgTime(features_train)
	
	for i,predict in enumerate(y_rbf):
		time=getTime(features_test[i])
		weighting=1-(avgTime-time)/avgTime
		weighting=math.sqrt(math.sqrt(math.sqrt((weighting+weighting)/2)))
		#靠近最後時間點的時間窗格，受當前交通時間影響的程度較大
		#因此在後面一個小時的預測時段，不使用當前交通時間與平均時間的加權運算(weighting=1)
		if(mn_time==2 or mn_time==4):
			weighting=1
		predict=predict*weighting
		printResult(fw,labels_test[i],predict,mn_time)

def main():
	loadWeather('weather (table 7)_test2')
	fw=open(predict_file_name, 'w')
	fw.writelines("intersection_id,tollgate_id,time_window,avg_travel_time"+"\n")
	for index,file in enumerate(files):
		global ID,TID
		ID=file.split("-")[0]
		TID=file.split("-")[1]

		fr = open(prepared_data_path + file+"_morning_training2_fixed.csv", 'r')
		lines_train = fr.readlines()
		fr1 = open(prepared_data_path + file+"_morning_training2_fixed.csv", 'r')
		lines_test_train = fr1.readlines()
		fr_test = open(prepared_data_path + file+"_morning_test2.csv", 'r')
		lines_test = fr_test.readlines()
		mn_time=1
		SVR_Model(fw,lines_train,lines_test_train,lines_test,mn_time)
		mn_time=2
		SVR_Model(fw,lines_train,lines_test_train,lines_test,mn_time)
		
		fr = open(prepared_data_path + file+"_night_training2_fixed.csv", 'r')
		lines_train = fr.readlines()
		fr1 = open(prepared_data_path + file+"_night_training2_fixed.csv", 'r')
		lines_test_train = fr1.readlines()
		fr_test = open(prepared_data_path + file+"_night_test2.csv", 'r')
		lines_test = fr_test.readlines()
		mn_time=3
		SVR_Model(fw,lines_train,lines_test_train,lines_test,mn_time)
		mn_time=4
		SVR_Model(fw,lines_train,lines_test_train,lines_test,mn_time)

if __name__ == '__main__':
	main()
