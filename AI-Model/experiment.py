#
# @author - Zachary Harris
#

from tictac_methods import *
import time, logging, pickle, itertools, random
from tqdm import tqdm

logger = logging.getLogger("Main")
logging.basicConfig(level=logging.DEBUG)


def main():
	vals = {"BoardRows":6, "BoardCols":5, "NumRows": 5, "NumCols": 4, "OutOfBounds":2, "Empty":0, "x":-1, "o":1}
	unfinished_Board = [[0 for _ in range(vals["BoardCols"] + 1)] for __ in range(vals["BoardRows"] + 1)]
	InitBoard(unfinished_Board, vals=vals)
	original_Board = copy.deepcopy(unfinished_Board)

	Board = copy.deepcopy(unfinished_Board)

	ShowBoard(Board, vals=vals)

	zeros_Board = [[j if j==2 else 0 for j in i] for i in Board]
	create_base_dict(zeros_Board)
	# print(zeros_Board)
	# test = list(itertools.combinations([(i//4+1,i%4+1) for i in range(20)], 8) )
	# print(list(itertools.combinations([i for i in range(8)], 4)  ))
	# print(test)
	# print(len(test))
	# out = [(i, int(random.random()*100)) for i in range(10)]
	# print(out)
	# print(sorted(out, reverse=True))
	# for i in range(100000000):
		# test = i % 3


# def __create_keys__(new_Board): #O(n^2) ~ 490 Trillion
# 	out = []
# 	for Player in [-1, 1]:
# 		Board = copy.deepcopy(new_Board)
# 		frontier = GetMoves(Player, Board)
# 		current_player = Player
# 		while frontier != []:
# 			current_state = frontier.pop(0)
# 			if not (current_state in out):
# 				out.append(current_state)
# 				current_player *= -1
# 				for new_key in GetMoves(current_player, out[0]):
# 					frontier.append(new_key)


def __create_keys_opti__(zeros_Board):
	# initial_p_locations = GetMoves(new_Board)
	# Permutations loop
	# outer_combs = list(itertools.combinations([(i//4+1,i%4+1) for i in range(20)], 8) )
	inner_combs = list(itertools.combinations([i for i in range(8)], 4))
	n_range = range(8)
	keys = [None for _ in range(125970*len(inner_combs))]
	vals = [0 for _ in range(len(keys))]
	count = 0
	for p_locations in tqdm( list(itertools.combinations([(i//4+1,i%4+1) for i in range(20)], 8)) ):  # O(n) ~ 140 mil
		for orientation in inner_combs:
			t_Board = copy.deepcopy(zeros_Board)
			for i in n_range:
				t_Board[p_locations[i][0]][p_locations[i][1]] = 1 if i in orientation else -1
			w_1  = Win( 1, t_Board)
			w_n1 = Win(-1, t_Board)
			if w_1 and w_n1:
				continue
			elif w_1:
				vals[i] = 1
			elif w_n1:
				vals[count] = -1
			else:
				vals[count] = 0

			keys[count] = t_Board
			count += 1

	keys = [i for i in keys if not (i is None)]
	vals = [vals[i] for i in range(len(keys))]
	return keys, vals
			
def create_base_dict(zeros_Board):
	keys, vals = __create_keys_opti__(zeros_Board)
	dict_out = dict.fromkeys(keys)
	# Add all winning states to the visited with val 1.
	for i in keys:
		dict_out[i] = vals[i]
	save_dict(dict_out)


def create_dict(zeros_Board):
	dict_out = load_dict()

	# # Add all 1-prior to visited with val 1. (Basically the step into a winning state)
	visited = [(i, dict_out[i]) for i in dict_out if abs(dict_out[i]) == 1]
	for (board, util) in visited:
		for b in GetBoardMoves(util, board):
			if not Win(util, t_Board):
				visited.append((b, util))
				dict_out[b] = util

	# Add all other states to frontier.
	frontier = sorted([(sum([1 if dict_out[ch] != 0 else 0 for ch in GetBoardMoves(1, b)]), b) for b in dict_out.keys() if dict_out[b] == 0], reverse=True)
	len_dict = len(dict_out.keys())
	while len(visited) < len_dict:
		quit()
		count = 0
		val = frontier[0][0]
		while frontier[count][0] == val:
			# if one child state has utility of 1,			O(1*n)
			# 	set utility to 1
			# 	add to visited
			# 	continue
			# if all child states have been visited,			O(1*n)
			# 	set utility to best out of children (relative to player).
			# 	add to visited
			# 	continue
			# else,											O(16^5 * n ~ n)
			# 	for all non-visited child states:
			# 		make utility the probability of reaching nearest goal(s) of its children (opponent or yours, 
			# 			multiple if equal depth) if opponent plays randomly (non-winning states have score 0) unless they have winning state.
			# 	(basically this is a depth-5 Breadth First Search at most for every non-visited node)

			# 	Combine all utilities
			# 	add to visited
			# 	continue
			count += 1
		frontier = sorted([(sum([1 if dict_out[ch] != 0 else 0 for ch in GetBoardMoves(1, b)]), b) for b in dict_out.keys() if dict_out[b] == 0], reverse=True)
	


def save_dict(dict_out, fname='./base_dict.pkl'):
	with open(fname, 'wb') as file:
		pickle.dump(dict_out, file)

def load_dict(fname='./base_dict.pkl'):
	with open(fname, 'wb') as file:
		dict_out = pickle.load(fname)
	return dict_out



if __name__ == "__main__":
	start_time = time.perf_counter()
	main()
	end_time = time.perf_counter()
	print('Total runtime: %.1f' % (end_time - start_time))
	time.sleep(.1)
	logger.debug('Total runtime: %.1f' % (end_time - start_time))





'''
Methodology:


Create a dictionary of all attainable board states. 

Dict: {Board_state: utility} (assume always playing 1's, and multiply board later to correct)

Notes:
	Board_state in dict is abs_value, allowing symmetry for players


Add all winning states to the visited with val 1.
Add all 1-prior to visited with val 1.
Add all 2-prior to visited with val -1 (i.e. for opponent).
Add all other states to frontier (state, children, num_visited).

For all states in frontier for player, sorted by ratio of children already visited:		O(n^2 logn)
	if one child state has utility of 1,			O(1*n)
		set utility to 1
		add to visited
		continue
	if all child states have been visited,			O(1*n)
		set utility to best out of children (relative to player).
		add to visited
		continue
	else,											O(16^5 * n ~ n)
		for all non-visited child states:
			make utility the probability of reaching nearest goal(s) of its children (opponent or yours, 
				multiple if equal depth) if opponent plays randomly (non-winning states have score 0) unless they have winning state.
		(basically this is a depth-5 Breadth First Search at most for every non-visited node)

		Combine all utilities
		add to visited
		continue


Calc all values for frontier as 1, visiting each dependency IFF it is also in the frontier.
Cycles (aka visited nodes) return 1 (max val)

'''