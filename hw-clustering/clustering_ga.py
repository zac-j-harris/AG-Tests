import numpy as np
import random, copy, pickle

def get_data(fname, subsample=None):
	arr = list(np.loadtxt(fname, delimiter=','))
	if not (subsample is None):
		arr = arr[:int(len(arr)*subsample)]
	return arr, len(arr[0])

def euclidian_dist(p1, p2):
	return (sum([(p2[i] - p1[i])**2 for i in range(len(p1))])) ** 0.5


def squared_error(p1, p2):
	return euclidian_dist(p1, p2)**2 


def get_fitness(population, arr):
	# smthg * (pop_size / num_clustroids)
	pop_size = len(population)
	fitness = [0.0 for _ in range(pop_size)]
	for p_i in range(pop_size):
		set_clustroids = population[p_i]
		num_clustroids = len(set_clustroids)
		m_mse = sum([ min([squared_error(clust, p) for clust in set_clustroids]) for p in arr]) / len(arr)
		
		# fitness[p_i] = (1.0 / m_mse) * (pop_size / num_clustroids)
		fitness[p_i] = (1.0 / m_mse)

	return fitness


def combine_parents(p1, p2):
	p_out = copy.deepcopy(p1)
	n_dim = len(p1[0])
	for c in range(len(p1)):
		if c >= len(p2):
			break
		p_out[c] = [(p1[c][d] + p2[c][d]) / 2.0 for d in range(n_dim)]
	return p_out


def crossover(population, fitness, elites, fit_cutoff):
	# print("Breeding parents")
	pop_size = len(population)
	new_pop = copy.deepcopy(population)
	for p_i in range(pop_size):
		if not (p_i in elites):
			p_2 = random.randint(0, pop_size-1)
			while fitness[p_2] < fit_cutoff and p_2 != p_i:
				p_2 = random.randint(0, pop_size-1)
			if fitness[p_2] > fitness[p_i]:
				p_1 = p_2
				p_2 = p_i
			else:
				p_1 = p_i
			new_pop[p_i] = combine_parents(new_pop[p_1], new_pop[p_1])
		# else:
			# print(fitness[p_i])
	return new_pop


def get_elites(fitness, elitism, selection_rate):
	# print("Gathering elites")
	elites = [-1 for _ in range(int(len(fitness) * elitism))]
	fit_cutoff = max(fitness)+1
	kept_count = 0
	count = 0
	while kept_count < int(len(fitness) * selection_rate):
		temp = 0
		for i in range(len(fitness)):
			if fitness[i] < fit_cutoff and fitness[i] > fitness[temp]:
				temp = i
		if count < len(elites):
			elites[count] = temp
			count += 1
		fit_cutoff = fitness[temp]
		kept_count += 1
	return elites, fit_cutoff


def get_mutation_delta(mutation_perc, m_fit):
	return (((random.random()*2.0) - 1.0) * (mutation_perc / min(m_fit, 1000) ))


def mutate(population, elites, mutation_rate, mutation_perc, struct_mut, m_fit, max_dim):
	new_pop = copy.deepcopy(population)
	for p_i in range(len(new_pop)):
		if not (p_i in elites):
			if random.random() < mutation_rate:
				if random.random() >= struct_mut:
					# Mutates new clustroid
					new_num_clust = max(1, int(len(new_pop[p_i]) + len(new_pop[p_i]) * get_mutation_delta(mutation_perc, m_fit)))
					temp = [[] for _ in range(new_num_clust)]
					for i in range(new_num_clust):
						if i < len(new_pop[p_i]):
							temp[i] = new_pop[p_i][i]
						else:
							temp[i] = [random.random() * d for d in max_dim]
				else:
					temp = copy.deepcopy(new_pop[p_i])
					temp = [[d * get_mutation_delta(mutation_perc, m_fit) for d in c] for c in temp]
				new_pop[p_i] = temp
	return new_pop


def run_ga(fname, f_out, population_size=30, elitism=0.1, mutation_rate=0.3, mutation_perc=0.15, struct_mut=0.3, selection_rate=0.9, subsample=0.10, min_generations=50):

	arr, dim = get_data(fname, subsample=subsample)
	clustroid_num = [1 for _ in range(population_size)]
	max_dim = [max([i[d] for i in arr]) for d in range(dim)]
	population = [[[random.random() * d for d in max_dim] for c_i in range(c)] for c in clustroid_num]
	# [pop_size][num_clustroids][num_dim]
	# [individual][clustroid][dimension]

	fitness = get_fitness(population, arr)
	elites, fit_cutoff = get_elites(fitness, elitism, selection_rate)
	population = crossover(population, fitness, elites, fit_cutoff)
	population = mutate(population, elites, mutation_rate, mutation_perc, struct_mut, max(fitness), max_dim)
	avg_fit = (sum(fitness) / len(fitness)) + 1
	generation = 1
	while avg_fit > (sum(fitness) / len(fitness)) or generation < min_generations:
		# if generation % 10 == 0:
		print("Generation: %s" % generation, max(fitness))
		generation += 1
		avg_fit = (sum(fitness) / len(fitness))
		fitness = get_fitness(population, arr)
		elites, fit_cutoff = get_elites(fitness, elitism, selection_rate)
		# print('elites gathered')
		population = crossover(population, fitness, elites, fit_cutoff)
		# print('Parents Bred')
		population = mutate(population, elites, mutation_rate, mutation_perc, struct_mut, max(fitness), max_dim)

	# print(fitness)
	max_fit = 0
	for i in range(len(fitness)):
		if fitness[i] > fitness[max_fit]:
			max_fit = i
	dict_out = {'Fitness': fitness[max_fit], 'Clustroids': population[max_fit]}

	with open(f_out, 'wb') as file:
		pickle.dump(dict_out, file)




def main():
	# quit()
	home_dir = '/home/zharris1/Documents/Github/Arcc-git-tests/hw-clustering/'
	fnames = ['Harris_array_1M.csv', 'Harris_array_50K.csv']
	f_outs = ['Harris_1M_out.pkl', 'Harris_50K_out.pkl']
	
	for i in range(2):
		run_ga(fname=home_dir + fnames[i], f_out=home_dir + f_outs[i], subsample=None)

	# print(get_data('Harris_array_1M.csv'))
	# save_array(ar1_1, fname="Harris_array_1M_labels.csv")
	# save_array(ar1_2, fname="Harris_array_1M.csv")
	# save_array(ar2_1, fname="Harris_array_50K_labels.csv")
	# save_array(ar2_2, fname="Harris_array_50K.csv")






if __name__=="__main__":
	main()