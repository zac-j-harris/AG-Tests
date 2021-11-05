#
# @author - Zachary Harris
#

from tictac_methods import *
import time, logging, pickle, itertools, random, lzma
from tqdm import tqdm


def get_util_old(init_MoveList):

	def check_movelist(MoveList):
		vals = [0 for _ in MoveList]
		count = 0
		for board_i in MoveList:
			if Win(1, board_i):
				vals[count] = 1
			count += 1
		return vals
	value = check_movelist(init_MoveList)
	if max(value) > 0:
		return value

	def bfs(queue):
		vals = check_movelist(queue)
		if max(vals) > 0:
			return [-1.0 * i for i in vals]
		vals = bfs([j for b in queue for j in GetBoardMoves(change_player(b))])
		num_condense = [len(GetBoardMoves(change_player(b))) for b in queue]
		return [-1.0* sum(vals[i+sum(num_condense[:i]):i+sum(num_condense[:i+1])])/(num_condense[i]) for i in range(len(queue))]
		# return [-1.0 * i for i in vals]
		# return [-1.0*i for i in bfs([j for b in queue for j in GetBoardMoves(change_player(b))])]

	q = [j for b in init_MoveList for j in GetBoardMoves(change_player(b))]
	# value = [-1.0 * i for i in bfs(q)]
	value = bfs(q)
	num_condense = [len(GetBoardMoves(change_player(b))) for b in q]
	value = [-1.0* sum(value[i+sum(num_condense[:i]):i+sum(num_condense[:i+1])])/(num_condense[i]) for i in range(len(q))]
	# value = [sum(value)]
	# return sum(value) / len(value), newfound_depth + 1
	return value



# min_MonteCarlo_trials = 5000 # 1000
max_MonteCarlo_trials = 500 # 1000
# run_MonteCarlo_sims = True
MaxDepth = 30 # if run_MonteCarlo_sims else 2 # goes by 2, so 5 is actually a search depth of 10
# num_playthroughs = 1
# pruning = True

# RUNTIME_COUNTER = 0
relative_moves = [(-1,0), (0,-1), (0,1), (1,0)]
# MAX_DEPTH = 0



def get_util(Player, MoveList, dict_vals, depth=1, sim=False):
	# global RUNTIME_COUNTER, MAX_DEPTH
	value = [0 for _ in MoveList]
	# MAX_DEPTH = depth * 2 - 1 if MAX_DEPTH < depth * 2 - 1 else MAX_DEPTH

	sim_i1 = random.randint(0, len(MoveList)-1) if sim else 0
	for move_i in range(len(MoveList)):
		move_i = sim_i1 if sim else move_i
		# move = MoveList[move_i]
		# board_i = ApplyMove(Board, move)
		# board_i = MoveList[move_i]

		if dict_vals[str(MoveList[move_i])] != -10:
			value[move_i] = dict_vals[str(MoveList[move_i])]
			continue
		elif Win(Player, MoveList[move_i]):
			# value[move_i] = 1 / depth
			value[move_i] = 1
			break
		else:
			# op_moves = GetMoves(Player * -1, board_i)
			op_moves = GetBoardMoves(1, change_player(MoveList[move_i]))
			num_trials = 1
			if not sim: # if not in middle of MC simulation
				num_trials = max_MonteCarlo_trials
			try:
				op_moves = [random.choice(op_moves) for i in range(num_trials)]
			except:
				continue

			num_trials = 0
			for op_move_i in range(len(op_moves)):
				num_trials += 1
				# MAX_DEPTH = depth * 2 if MAX_DEPTH < depth * 2 else MAX_DEPTH
				# RUNTIME_COUNTER += 20
				board_ij = op_moves[op_move_i]
				# MiniMax, with alpha-beta pruning
				# board_ij = ApplyMove(board_i, op_move)
				if dict_vals[str(board_ij)] != -10:
					value[move_i] = -1.0 * dict_vals[str(board_ij)]
					continue 
				elif Win(1, board_ij):     # Opposition wins, prune
					# value[move_i] = -1 / depth
					value[move_i] = -1
					break
				elif depth < MaxDepth:
					# minimax = get_util(Player, GetBoardMoves(1, change_player(board_ij)), depth=depth + 1, sim=True)
					# value[move_i] += minimax / depth
					value[move_i] += get_util(Player, GetBoardMoves(1, change_player(board_ij)), depth=depth + 1, sim=True)
					# else:
					# 	value[move_i] = minimax if (minimax < value[move_i]) or (value[move_i] == 0) else value[move_i]
					# if (value[move_i] <= max(value) or value[move_i] < 0) and op_move_i >= min_MonteCarlo_trials and pruning:
					# 	# Minimal value less than other branch min, prune
					# 	break
			# if not sim:
			value[move_i] = value[move_i] / num_trials
		if sim:
			break

	if depth == 1:
		return value
	else:
		return sum(value) / len(value)



def get_utility(x_in):
	# cur_board, dict_out = x_in
	child_boards = GetBoardMoves(1, change_player(get_list_board(x_in[0])))
	child_utils = [x_in[1][str(b)] for b in child_boards]

	if all([i != -10 for i in child_utils]):
		# current_util = min(child_utils) * -1.0
		return max(child_utils) * -1.0
		# visited.append((frontier[count], current_util))
	else:
		# val = get_util(Player=1, MoveList=[get_list_board(frontier[count][1])], dict_vals=dict_out)
		# val = val if type(val) != list else val[0]
		vis_child = [None if child_utils[i] == -10 else child_boards[i] for i in range(len(child_boards))]
		n_vis_child = [None if child_utils[i] != -10 else child_boards[i] for i in range(len(child_boards))]
		utils = [0 if child_utils[i] == -10 else -1.0 * child_utils[i] for i in range(len(child_boards))]

		n_vis_childs_reduced = [i for i in n_vis_child if not (i is None)]

		# vals = get_util(n_vis_childs_reduced)
		vals = get_util(Player=1, MoveList=n_vis_childs_reduced, dict_vals=x_in[1])

		temp_val_counter = 0
		for i in range(len(utils)):
			if vis_child[i] is None:
				utils[i] = -1.0 * vals[temp_val_counter]
				temp_val_counter += 1
		return min(utils)  # Minimized because child would pick their best