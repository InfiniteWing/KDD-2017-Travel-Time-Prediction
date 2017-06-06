#############################################################
#															#
#	time_data_pre.py										#
#	提取資料中預測前一小時各道路的時間花費特徵以供模型學習	#
#															#
#															#
#############################################################

# import necessary modules
import math
from datetime import datetime,timedelta

file_suffix = '.csv'
path = "data\\"								#原始訓練資料路徑
prepared_data_path="prepared_data\\"		#特徵提取完成的資料路徑
train_time_windows=[7,16]					#會用到的訓練時間
predict_time_windows=[8,9,17,18]			#會用到的目標時間
use_time_windows=train_time_windows+predict_time_windows

#各route所涵蓋的link
link_route={}
link_route["A-2"]="110,123,107,108,120,117"
link_route["A-3"]="110,123,107,108,119,114,118,122"
link_route["B-1"]="105,100,111,103,116,101,121,106,113"
link_route["B-3"]="105,100,111,103,122"
link_route["C-1"]="115,102,109,104,112,111,103,116,101,121,106,113"
link_route["C-3"]="115,102,109,104,112,111,103,122"

link_time={}
time_train={}
time_predict={}
time_check={}
test_time_train={}
test_time_check={}
weathers={}
rainingTotalTime={}

#讀取氣候資料
def loadWeather(in_file):

	out_suffix = ''
	in_file_name = in_file + file_suffix
	out_file_name = in_file.split('_')[1] + out_suffix + file_suffix

	fr = open(path + in_file_name, 'r')
	fr.readline()  # skip the header
	vol_data = fr.readlines()
	fr.close()
	
	for i in range(len(vol_data)):
		each_pass = vol_data[i].replace('"', '').split(',')
		precipitation = float(each_pass[8])
		hour=int(each_pass[1])
		time_window = each_pass[0]
		time_window = datetime.strptime(time_window, "%Y/%m/%d")
		time_window_minute = int(math.floor(time_window.minute / 20) * 20)
		time = datetime(time_window.year, time_window.month, time_window.day,
									 time_window.hour, time_window_minute, 0)
		if time.month*31+time.day not in weathers:
			weathers[time.month*31+time.day] = {}
		if(hour>0):
			for h in range(3):
				weathers[time.month*31+time.day][hour-h-1]=precipitation

#取得一個Link在指定時間窗格中，通過所需花費的平均時間
def getLinkAvgTime(link_time_window):
	avg_tt=sum(link_time_window)/float(len(link_time_window))
	variance=0
	for t in link_time_window:
		variance+=(t-avg_tt)*(t-avg_tt)
	variance=variance/float(len(link_time_window))
	sd=math.sqrt(variance)
	new_time=[]
	for t in link_time_window:
		if(abs(avg_tt-t)<=sd*3):
			new_time.append(t)
	if(len(new_time)==0):
		return avg_tt
	else:
		return sum(new_time)/float(len(new_time))
		
#取得完成一個route所花費的平均時間
def getRouteAvgTimeFromLink(route,time,link_time):
	try:
		use_seconds=0
		for link_id in link_route[route].split(','):
			use_seconds+=getLinkAvgTime(link_time[link_id][time.month*31+time.day][time.hour])
		return use_seconds
	except:
		return -1

#取得一個route中，各Link所花費的平均時間。
#作為特徵之一
def getLinkAvgTimeStr(route,time,link_time):
	try:
		use_hour=0
		if(time.hour<10):
			use_hour=7
		else:
			use_hour=16
		use_seconds_str=[]
		for link_id in link_route[route].split(','):
			use_seconds_str.append(str(getLinkAvgTime(link_time[link_id][time.month*31+time.day][use_hour])))
			use_seconds_str.append(str(len(link_time[link_id][time.month*31+time.day][use_hour])))
		return use_seconds_str
	except:
		return []

#建立訓練特徵
def prepareTrainingDatas(in_file):

	out_suffix = ''
	in_file_name = in_file + file_suffix
	out_file_name = in_file.split('_')[1] + out_suffix + file_suffix

	# Step 1: Load trajectories
	fr = open(path + in_file_name, 'r')
	fr.readline()  # skip the header
	traj_data = fr.readlines()
	fr.close()
	# 建立各Link時間的dictionary
	for i in range(24):
		link_time[str(i+100)]={}
	# Step 2: Create a dictionary to store travel time for each route per time window
	travel_times = {}  # key: route_id. Value is also a dictionary of which key is the start time for the time window and value is a list of travel times
	for i in range(len(traj_data)):
		each_traj = traj_data[i].replace('"', '').split(',')
		intersection_id = each_traj[0]
		tollgate_id = each_traj[1]

		route_id = intersection_id + '-' + tollgate_id
		if route_id not in travel_times.keys():
			travel_times[route_id] = {}

		trace_start_time = each_traj[3]
		travel_seq = each_traj[4]
		trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
		time_window_minute = math.floor(trace_start_time.minute / 20) * 20
		start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
									 trace_start_time.hour, time_window_minute, 0)
		time=start_time_window
		tt = float(each_traj[-1])
		# 國慶不管，因道路狀態不一樣
		if(time.month==10 and time.day in [1,2,3,4,5,6,7]):
			continue
		# 中秋節不管，因道路狀態不一樣
		if(time.month==9 and time.day in [15,16,17]):
			continue
		if start_time_window not in travel_times[route_id].keys():
			travel_times[route_id][start_time_window] = [tt]
		else:
			travel_times[route_id][start_time_window].append(tt)
		if(time.hour in use_time_windows):
			try:
				#依序解析該筆紀錄於各link所花費的時間
				for each_seq in travel_seq.split(';'):
					each_seq=each_seq.split('#')
					link_id=each_seq[0]
					start_datetime=each_seq[1]
					use_seconds=float(each_seq[2])
					if(time.month*31+time.day not in link_time[link_id]):
						link_time[link_id][time.month*31+time.day]={}
					if(time.hour not in link_time[link_id][time.month*31+time.day]):
						link_time[link_id][time.month*31+time.day][time.hour]=[]
					link_time[link_id][time.month*31+time.day][time.hour].append(use_seconds)
			except:
				pass

				
	for route in travel_times.keys():
		time_train[route]={}
		time_predict[route]={}
		route_time_windows = list(travel_times[route].keys())
		route_time_windows.sort()
		for time_window_start in route_time_windows:
			time=time_window_start
			if(time.hour in use_time_windows):
				time_window_end = time_window_start + timedelta(minutes=20)
				tt_set = travel_times[route][time_window_start]
				# 將資料分類成訓練(6-7,15-16)(採用各route中link平均加總)
				# 及預測(8-9,17-18)(採用原始資料中的route平均時間)
				# 用來作為主要特徵
				if(time.hour in predict_time_windows):
					avg_tt = round(sum(tt_set) / float(len(tt_set)), 2)
					if(time.month*31+time.day not in time_predict[route]):
						time_predict[route][time.month*31+time.day]={}
					if(time.hour not in time_predict[route][time.month*31+time.day]):
						time_predict[route][time.month*31+time.day][time.hour]=[]
					time_predict[route][time.month*31+time.day][time.hour].append(avg_tt)
				elif(time.hour in train_time_windows):
					avg_tt = getRouteAvgTimeFromLink(route,time,link_time)
					if(avg_tt==-1):
						continue
					if(time.month*31+time.day not in time_train[route]):
						time_train[route][time.month*31+time.day]={}
					if(time.hour not in time_train[route][time.month*31+time.day]):
						time_train[route][time.month*31+time.day][time.hour]=[]
					time_train[route][time.month*31+time.day][time.hour].append(avg_tt)
					
	for route in travel_times.keys():
		fw_m = open(prepared_data_path+route+"_morning_"+out_file_name, 'w')
		fw_n = open(prepared_data_path+route+"_night_"+out_file_name, 'w')
		route_time_windows = list(travel_times[route].keys())
		route_time_windows.sort()
		time_check[route]={}
		for time_window_start in route_time_windows:
			time=time_window_start
			if(time.hour in train_time_windows):
				if(time.month*31+time.day not in time_check[route]):
					time_check[route][time.month*31+time.day]=1
					for ttt in predict_time_windows:
						try:
							ww=weathers[time.month*31+time.day][ttt]
						except:
							#當時的天氣資料有遺漏，不採用
							continue
						try:
							time = datetime(time.year, time.month, time.day,
									 ttt, time.minute, 0)
							link_str=getLinkAvgTimeStr(route,time,link_time)
							if(len(link_str)==0):
								continue
							if(ttt<10):
								avg_tt=sum(time_train[route][time.month*31+time.day][7])/len(time_train[route][time.month*31+time.day][7])
								for label in time_predict[route][time.month*31+time.day][ttt]:
									out_line=','.join([str(label)]+link_str+[str(avg_tt)])
									fw_m.writelines(out_line+"\n")
							else:
								avg_tt=sum(time_train[route][time.month*31+time.day][16])/len(time_train[route][time.month*31+time.day][16])
								for label in time_predict[route][time.month*31+time.day][ttt]:
									out_line=','.join([str(label)]+link_str+[str(avg_tt)])
									fw_n.writelines(out_line+"\n")
						except:
							pass
		fw_m.close()
		fw_n.close()

#建立測試特徵
def prepareTestingDatas(in_file):

	out_suffix = ''
	in_file_name = in_file + file_suffix
	out_file_name = in_file.split('_')[1] + out_suffix + file_suffix

	# Step 1: Load trajectories
	fr = open(path + in_file_name, 'r')
	fr.readline()  # skip the header
	traj_data = fr.readlines()
	fr.close()
	#print(traj_data[0])
	# 建立各Link時間的dictionary
	for i in range(24):
		link_time[str(i+100)]={}
	# Step 2: Create a dictionary to store travel time for each route per time window
	travel_times = {}  # key: route_id. Value is also a dictionary of which key is the start time for the time window and value is a list of travel times
	for i in range(len(traj_data)):
		each_traj = traj_data[i].replace('"', '').split(',')
		intersection_id = each_traj[0]
		tollgate_id = each_traj[1]

		route_id = intersection_id + '-' + tollgate_id
		if route_id not in travel_times.keys():
			travel_times[route_id] = {}

		trace_start_time = each_traj[3]
		travel_seq=each_traj[4]
		trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
		time_window_minute = math.floor(trace_start_time.minute / 20) * 20
		start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
									 trace_start_time.hour, time_window_minute, 0)
		tt = float(each_traj[-1]) # travel time

		if start_time_window not in travel_times[route_id].keys():
			travel_times[route_id][start_time_window] = [tt]
		else:
			travel_times[route_id][start_time_window].append(tt)
		time=start_time_window
		if(time.hour in use_time_windows):
			try:
				for each_seq in travel_seq.split(';'):
					each_seq=each_seq.split('#')
					link_id=each_seq[0]
					start_datetime=each_seq[1]
					use_seconds=float(each_seq[2])
					if(time.month*31+time.day not in link_time[link_id]):
						link_time[link_id][time.month*31+time.day]={}
					if(time.hour not in link_time[link_id][time.month*31+time.day]):
						link_time[link_id][time.month*31+time.day][time.hour]=[]
					link_time[link_id][time.month*31+time.day][time.hour].append(use_seconds)
			except:
				pass
			
	for route in travel_times.keys():
		test_time_train[route]={}
		route_time_windows = list(travel_times[route].keys())
		for time_window_start in route_time_windows:
			time=time_window_start
			if(time.hour in use_time_windows):
				tt_set = travel_times[route][time_window_start]
				avg_tt = round(sum(tt_set) / float(len(tt_set)), 2)
				avg_tt=getRouteAvgTimeFromLink(route,time,link_time)
				# 將資料分類成訓練(6-7,15-16)
				# 及預測(8-9,17-18)
				# 用來製作成線性回歸需要的資料
				if(time.hour in train_time_windows):
					if(time.month*31+time.day not in test_time_train[route]):
						test_time_train[route][time.month*31+time.day]={}
					if(time.hour not in test_time_train[route][time.month*31+time.day]):
						test_time_train[route][time.month*31+time.day][time.hour]=[]
					test_time_train[route][time.month*31+time.day][time.hour].append(avg_tt)
	for route in travel_times.keys():
		fw_m = open(prepared_data_path+route+"_morning_"+out_file_name, 'w')
		fw_n = open(prepared_data_path+route+"_night_"+out_file_name, 'w')
		route_time_windows = list(travel_times[route].keys())
		route_time_windows.sort()
		test_time_check[route]={}
		for time_window_start in route_time_windows:
			time=time_window_start
			if(time.month*31+time.day not in test_time_check[route]):
				test_time_check[route][time.month*31+time.day]=1
				for ttt in train_time_windows:
					try:
						if(len(test_time_train[route][time.month*31+time.day][ttt])<1):
							test_time_train[route][time.month*31+time.day][ttt].append(0)
					except:
						#該時段沒有紀錄，避免程式異常崩潰需做處理
						test_time_train[route][time.month*31+time.day][ttt]=[]
						test_time_train[route][time.month*31+time.day][ttt].append(0)
						
				#在訓練資料中的目標標籤欄位被測試的日期所取代，用以輸出正確的預測格式
				time = datetime(time.year, time.month, time.day,
									 8, time.minute, 0)
				link_str=getLinkAvgTimeStr(route,time,link_time)
				avg_tt=sum(test_time_train[route][time.month*31+time.day][7])/len(test_time_train[route][time.month*31+time.day][7])
				out_line=','.join([time.strftime("%d")]+link_str+[str(avg_tt)])
				fw_m.writelines(out_line+"\n")
				time = datetime(time.year, time.month, time.day,
									 17, time.minute, 0)
				link_str=getLinkAvgTimeStr(route,time,link_time)
				avg_tt=sum(test_time_train[route][time.month*31+time.day][16])/len(test_time_train[route][time.month*31+time.day][16])
				out_line=','.join([time.strftime("%d")]+link_str+[str(avg_tt)])
				fw_n.writelines(out_line+"\n")
		fw_n.close()
		fw_m.close()



def main():
	in_file = 'weather (table 7)_training2'
	loadWeather(in_file)
	in_file = 'trajectories(table 5)_training2'
	prepareTrainingDatas(in_file)
	test_in_file = 'trajectories(table 5)_test2'
	prepareTestingDatas(test_in_file)

if __name__ == '__main__':
	main()



