#
# @author - Zachary Harris
#
import copy

relative_moves = [(-1,0), (0,-1), (0,1), (1,0)]
NumRows = 5
NumCols = 4
Empty=0

def GetBoardMoves(Player, Board):
	p_locations = [(i, j) for i in range(1, NumRows+1) for j in range(1, NumCols+1) if Board[i][j] == Player]
	MoveList = [[i, j, i+m, j+n] for (i, j) in p_locations for m, n in relative_moves if Board[i + m][j + n] == Empty]
	# for (i, j) in p_locations:
	# 	#-------------------------------------------------------------
	# 	#  Check move directions (m,n) = (-1,0), (0,-1), (0,1), (1,0)
	# 	#-------------------------------------------------------------
	# 	for m, n in relative_moves:
	# 		if Board[i + m][j + n] == Empty:
	# 			MoveList.append([i, j, i+m, j+n])
	out = [ApplyMove(Board, move) for move in MoveList]
	return out


def InitBoard (Board, vals):
	#-------------------------------------------------------------------------
	# Initialize the game board.
	#-------------------------------------------------------------------------
	def is_odd(n):
		return n % 2 == 1

	for i in range(0,vals["BoardRows"]+1):
		for j in range(0,vals["BoardCols"]+1):
			Board[i][j] = vals["OutOfBounds"]

	for i in range(1,vals["BoardRows"]):
		for j in range(1,vals["BoardCols"]):
			Board[i][j] = vals["Empty"]

	for j in range(1,vals["BoardCols"]):
		if is_odd(j):
			Board[1][j] = vals["x"]
			Board[vals["NumRows"]][j] = vals["o"]
		else:
			Board[1][j] = vals["o"]
			Board[vals["NumRows"]][j] = vals["x"]


def ShowBoard (Board, vals):
	print("")
	row_divider = "+" + "-"*(vals["NumCols"]*4-1) + "+"
	print(row_divider)

	for i in range(1,vals["BoardRows"]):
		for j in range(1,vals["BoardCols"]):
			if Board[i][j] == vals["x"]:
				print('| X ',end="")
			elif Board[i][j] == vals["o"]:
				print('| O ',end="")
			elif Board[i][j] == vals["Empty"]:
				print('|   ',end="")
		print('|')
		print(row_divider)

	print("")


def change_player(board):
		return [[-1 * i if abs(i) == 1 else i for i in row] for row in board]


def ApplyMove (Board, Move, Empty=0):
	#-------------------------------------------------------------------------
	# Perform the given move, and update Board.
	#-------------------------------------------------------------------------

	# FromRow, FromCol, ToRow, ToCol = Move
	newBoard = copy.deepcopy(Board)
	newBoard[Move[2]][Move[3]] = Board[Move[0]][Move[1]]
	newBoard[Move[0]][Move[1]] = Empty
	return newBoard


def Win (Player, Board):
	#-------------------------------------------------------------------------
	# Determines if Player has won, by finding '3 in a row'.
	#-------------------------------------------------------------------------
	def get_diag(hor_c, vert_c, row, col):
		'''
			Returns True if there is a diagonal scoring in the given configuration on the board.
			Returns False otherwise.
		'''
		for i in range(3):
			if Board[row + (i*vert_c)][col + (i*hor_c)] != Player:
				return False
		return True

	for col in range(1,5):
		in_a_col = 0
		in_a_row = [i[col] + 1 if i[col] == Player else 0 for i in Board]
		for row in range(1,6):
			# Test vertical & horizontal scoring
			in_a_col = in_a_col + 1 if Board[row][col] == Player else 0
			if in_a_col == 3 or max(in_a_row) == 3:
				return True

			# Test diagonal scoring
			h_c = 1 if col < 3 else -1
			if row == 3:
				# Middle row, any column has 2 possible diagonals
				if get_diag(h_c, 1, row, col) or get_diag(h_c, -1, row, col):
					return True
			else:
				# All other rows only have a single diagonal
				v_c = 1 if row <= 3 else -1
				if get_diag(h_c, v_c, row, col):
					return True
	return False