#
# @author - Zachary Harris
#


'''
DONE: modify heuristic for unknown children to be original heuristic from my thing.
DONE: modify that to include known values for heuristic function (to shorten search)


'''





from tictac_methods import *
import time, logging, pickle, itertools, random, lzma
from tqdm import tqdm
from estimation_heuristic import *
import multiprocessing as mp

logger = logging.getLogger("Main")
logging.basicConfig(level=logging.DEBUG)


def main():
	vals = {"BoardRows":6, "BoardCols":5, "NumRows": 5, "NumCols": 4, "OutOfBounds":2, "Empty":0, "x":-1, "o":1}
	unfinished_Board = [[0 for _ in range(vals["BoardCols"] + 1)] for __ in range(vals["BoardRows"] + 1)]
	InitBoard(unfinished_Board, vals=vals)
	# unfinished_Board = [tuple([j for j in i]) for i in unfinished_Board]
	# print(unfinished_Board)
	original_Board = copy.deepcopy(unfinished_Board)

	Board = copy.deepcopy(unfinished_Board)

	ShowBoard(Board, vals=vals)

	# test = [str([j for j in range(i, i+11)]) for i in range(3)]
	# print(test)
	# test2 = dict.fromkeys(test)
	# print(test2)
	# for i in test2:
	# 	print(i)
	# 	print(type(i))
	# 	test2[i] = 1


	# zeros_Board = [[j if j==2 else 0 for j in i] for i in Board]
	# create_base_dict(zeros_Board)

	# def __create_test_dict__():
	# 	dict_out = load_dict()
	# 	keys = list(dict_out.keys())[:int(len(dict_out)*0.01)]
	# 	test_dict = dict.fromkeys(keys)
	# 	for k in test_dict:
	# 		test_dict[k] = dict_out[k]
	# 	save_dict(test_dict, './comp_test_dict.xz')
	# __create_test_dict__()


	create_weighted_dict()


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
				# t_Board[p_locations[i][0]] = tuple([ (1 if i in orientation else -1) if j==p_locations[i][1] else j for j in range(len(t_Board[p_locations[i][0]])) ])
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

	keys = [str(i) for i in keys if not (i is None)]
	vals = [vals[i] for i in range(len(keys))]
	return keys, vals
			

def create_base_dict(zeros_Board):
	keys, vals = __create_keys_opti__(zeros_Board)
	base_dict = dict.fromkeys(keys)
	# Add all winning states to the visited with val 1.
	for i in range(len(keys)):
		base_dict[keys[i]] = vals[i]
	save_dict(base_dict)



def create_weighted_dict():

	dict_out = load_dict()

	for k in dict_out:
		if dict_out[k] == 0:
			dict_out[k] = -10

	# # Add all 1-prior to visited with val 1. (Basically the step into a winning state)
	len_visited = sum( [1 for i in dict_out if dict_out[i] != 0] )

	# t = [(b, util) for (board, util) in visited for b in GetBoardMoves(util, get_list_board(board))]
	# for (board, util) in tqdm( visited ):
	# 

	for b, util in tqdm( [(ch_b, dict_out[board]) for board in dict_out if dict_out[board] != -10
	                      for ch_b in GetBoardMoves(dict_out[board], get_list_board(board))]):
		if not Win(util, b):
			len_visited += 1
			dict_out[str(b)] = util

	# len_dict = len(dict_out.keys())
	def pop_inds(l, indexes):
		popped = []
		for i in sorted(indexes, reverse=True):
			popped.append(l[i][1])
			del l[i]
		return l, popped

	def pop_top(l):
		m = max([i[0] for i in l])
		inds = []
		for i in range(len(l)):
			if l[i][0] == m:
				inds.append(i)
		return pop_inds(l, inds)

	frontier = [(sum([1 if dict_out[str(ch)] != -10 else 0 for ch in GetBoardMoves(1, get_list_board(b))]), b)
	            for b in tqdm( dict_out ) if dict_out[b] == -10]

	pbar = tqdm(total=len(frontier))
	print('Beginning secondary iteration.')

	while frontier != []:
		frontier, mp_boards = pop_top(frontier)
		count = len(mp_boards)
		pool = mp.Pool(mp.cpu_count())

		utilities = list(pool.map(get_utility, [(b, dict_out) for b in mp_boards]))
		pool.close()
		pbar.update(count)

		# each state in frontier is given utility as an end point
		# 	- i.e. given utility from opponent's moves
		for b_i in range(len(mp_boards)):
			cur_board = mp_boards[b_i]
			dict_out[cur_board] = utilities[b_i]
		len_visited += 1
		frontier = [(sum([1 if dict_out[str(ch)] != -10 else 0 for ch in GetBoardMoves(1, get_list_board(b))]), b)
		            for _, b in frontier]

	pbar.close()
	save_dict(dict_out, fname='./comp_weighted_dict.xz')


def save_dict(dict_out, fname='./comp_base_dict.xz'):
	with lzma.open(fname, "wb") as file:
		pickle.dump(dict_out, file)


def load_dict(fname='./comp_base_dict.xz'):
	with lzma.open(fname, "rb") as file:
		dict_out = pickle.load(file)
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