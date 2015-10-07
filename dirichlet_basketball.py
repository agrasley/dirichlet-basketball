import numpy
import pandas
from sklearn import mixture
import csv

FILENAME = "leagues_NBA_2015_per_minute_per_minute.csv"
STATS = pandas.read_csv(open(FILENAME,'rb'))

class PlayerGroup():

	def __init__(self,group_num):
		self.group_num = group_num
		self.player_list = []

	def get_player_names(self):
		return [STATS.irow(index).Player for index in self.player_list]

	def get_player_positions(self):
		positions = {
			"PG": 0,
			"PG-SG": 0,
			"SG-PG": 0,
			"SG": 0,
			"SG-SF": 0,
			"SF-SG": 0,
			"SF": 0,
			"SF-PF": 0,
			"PF-SF": 0,
			"PF": 0,
			"C": 0,
		}
		for index in self.player_list:
			positions[STATS.irow(index).Pos] += 1
		return positions

	def get_player_averages(self):
		averages = {}
		for col in ['Age','G','GS','MP','FG','FGA','3P','3PA','2P','2PA','FT','FTA','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS']:
			averages[col] = sum([STATS.irow(index)[col] for index in self.player_list]) / len(self.player_list)
		averages['FG%'] = averages['FG'] / averages['FGA']
		averages['3P%'] = averages['3P'] / averages['3PA']
		averages['2P%'] = averages['2P'] / averages['2PA']
		averages['FT%'] = averages['FT'] / averages['FTA']
		return averages

def output_to_files(groups,text_filename,csv_filename):
	with open(text_filename,'w') as out:
		for index,group in groups.items():
			out.write("Group "+str(index)+":\n\nPlayers: ")
			out.write(", ".join(group.get_player_names()))
			out.write("\n\nPositions:\n")
			positions = group.get_player_positions()
			for k,v in positions.items():
				out.write(k+"\t"+str(v)+"\n")
			out.write("\n\n")
	with open(csv_filename, 'w') as out:
		writer = csv.writer(out)
		writer.writerow(['Group','Age','G','GS','MP','FG','FGA','FG%','3P','3PA','3P%','2P','2PA','2P%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS'])
		for index,group in groups.items():
			averages = group.get_player_averages()
			writer.writerow([index,averages['Age'],averages['G'],averages['GS'],averages['MP'],averages['FG'],averages['FGA'],averages['FG%'],averages['3P'],averages['3PA'],averages['3P%'],averages['2P'],averages['2PA'],averages['2P%'],averages['FT'],averages['FT%'],averages['ORB'],averages['DRB'],averages['TRB'],averages['AST'],averages['STL'],averages['BLK'],averages['TOV'],averages['PF'],averages['PTS']])

def create_player_groups(np_array):
	groups = {}
	group_nums = np_array.tolist()
	group_set = set(group_nums)
	for num in group_set:
		groups[num] = PlayerGroup(num)
	for index,group_num in enumerate(group_nums):
		groups[group_num].player_list.append(index)
	return groups

def _output_results(np_array):
	groups = np_array.tolist()
	players = STATS['Player']
	return list(zip(players,groups))

def simple_stats_no_minutes():
	x = numpy.loadtxt(open(FILENAME, 'rb'), delimiter=",", usecols=(22,23,24,25,26,27,28),skiprows=1)
	dpgmm = mixture.DPGMM(n_iter=100,n_components=25)
	dpgmm.fit(x)
	return _output_results(dpgmm.predict(x))

def simple_stats_with_minutes():
	x = numpy.loadtxt(open(FILENAME, 'rb'), delimiter=",", usecols=(7,22,23,24,25,26,27,28),skiprows=1)
	dpgmm = mixture.DPGMM(n_iter=100,n_components=25)
	dpgmm.fit(x)
	return _output_results(dpgmm.predict(x))

def advanced_stats_only():
	x = numpy.loadtxt(open(FILENAME, 'rb'), delimiter=",", usecols=(7,11,12,14,15,17,18,20,21,23,24,25,26,27),skiprows=1)
	dpgmm = mixture.DPGMM(n_iter=100,n_components=25)
	dpgmm.fit(x)
	return create_player_groups(dpgmm.predict(x))

def all_relevant_stats():
	x = numpy.loadtxt(open(FILENAME, 'rb'), delimiter=",", usecols=(7,8,9,11,12,14,15,17,18,20,21,22,23,24,25,26,27,28),skiprows=1)
	dpgmm = mixture.DPGMM(n_iter=100,n_components=25)
	dpgmm.fit(x)
	return _output_results(dpgmm.predict(x))

