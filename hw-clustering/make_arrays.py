import numpy as np
import random


def save_array(array, fname):
	arr = np.asarray(array)
	print(arr[:10])
	# quit()
	# np.savetxt(fname, arr, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
	np.savetxt(fname, arr, delimiter=',', newline='\n', fmt='%.5f')


def euc_dist(x1, y1, x2, y2):
	return ((x2-x1)**2 + (y2-y1)**2)**0.5

def array_1_i(mat, x_in, y_in):
	if random.random() < 0.001:
		return (random.random()*30+0.5, random.random()*30+0.5, float(-1))
	
	if mat[y_in][x_in][0] == 'square':
		x = random.random() * 2 + float(x_in)
		y = random.random() * 2 + float(y_in)
	elif mat[y_in][x_in][0] == 'circle':
		x = random.random() * 2 + float(x_in)
		y = random.random() * 2 + float(y_in)
		while euc_dist(x, y, x_in+1.0, y_in+1.0) > 1.0:
			x = random.random() * 2 + float(x_in)
			y = random.random() * 2 + float(y_in)
	else:
		x = random.random() * 2 + float(x_in)
		y = random.random() * 2 + float(y_in)
		while euc_dist(x, y, x_in+1.0+mat[y_in][x_in][2], y_in+1.0+mat[y_in][x_in][1]) + euc_dist(x, y, x_in+1.0-mat[y_in][x_in][2], y_in+1.0-mat[y_in][x_in][1]) > 2 or x < x_in or x > x_in + 2 or y < y_in or y > y_in + 2:
			x = random.random() * 2 + float(x_in)
			y = random.random() * 2 + float(y_in)
	return (x, y, float(y_in*30+x_in+1))


def array_2_i(centers):
	i = random.randint(0, len(centers)-1)
	c_d = centers[i]
	if random.random() < 0.01:
		return [random.random()*200-100 if j > 0 else -1 for j in range(len(c_d)+1)]
	return [random.random()*10-5 + c_d[j-1] if j > 0 else float(i+1) for j in range(len(c_d)+1)]



def main():
	ar1_len = 1000000
	ar1_mat = [[random.choice([['square'], ['circle'], ('ellipse', (random.random()-0.5)*2**0.5, (random.random()-0.5)*2**0.5)]) for i in range(30)] for _ in range(30)]
	ar1 = [array_1_i(ar1_mat, int(random.random()*30), int(random.random()*30)) for _ in range(ar1_len)]
	ar1_1 = [i[2] for i in ar1]
	save_array(ar1_1, fname="Harris_array_1M_labels.csv")
	ar1_2 = [i[:2] for i in ar1]
	save_array(ar1_2, fname="Harris_array_1M.csv")


	ar2_len = 50000
	n_dim = 50
	centers = [[random.random()*200-100.0 for i in range(n_dim)] for _ in range(10*n_dim)]
	ar2 = [array_2_i(centers) for i in range(ar2_len)]
	ar2_1 = [i[0] for i in ar2]
	save_array(ar2_1, fname="Harris_array_50K_labels.csv")
	ar2_2 = [i[1:] for i in ar2]
	save_array(ar2_2, fname="Harris_array_50K.csv")

if __name__=="__main__":
	main()