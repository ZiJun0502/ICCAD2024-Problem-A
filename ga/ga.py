import random
import numpy as np
from queue import PriorityQueue
from os import mkdir
from os.path import join, exists
from utils.library import Library
from utils.config import Config
from utils.cost_interface import CostInterface
from DRiLLS.abcSession import abcSession
import time
def log(message: str, end='\n'):
    print(message, end=end)
    pass
class GA:
    def __init__(self, n=50, n_init=100, dim=8, dim_limit=[], bits=5, 
                 crossover_rate=0.9, mutation_rate=0.4, n_iter=20, k_solution=5,
                 design_path="./playground/design_preprocessed.v", init_population=[],
                 dir_suffix="ga_genlib"):
        self.dir_suffix = dir_suffix
        self.n = n
        self.n_init = n_init
        self.dim = dim
        self.dim_limit = dim_limit
        self.bits = bits
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_iter = n_iter
        self.k_solution = k_solution
        self.design_path = design_path
        self.config = Config()
        self.playground_dir = join(self.config.params['playground_dir'], dir_suffix)
        if not exists(self.playground_dir):
            mkdir(self.playground_dir)
        self.iteration_dir = self.playground_dir
        self.library = Library()
        self.abcSession = abcSession()
        self.cell_map = self.library.cell_map
        self.gate_types = self.library.gate_types
        self.population = self.init_population(init_population)

    def init_population(self, init_population):
        """
        Build a random population of size n_init, 
        and only preserve top-n candidates.
        """
        start = time.time()
        population = []
        population = [self.random_chromosome() for _ in range(self.n_init-len(init_population))]
        # seeded initialization
        if len(init_population):
            for cell_ids in init_population:
                init_chromosome = [self.encode_gene(i) for i in cell_ids]
                population += [init_chromosome]
        self.iteration_dir = join(self.playground_dir, f"init_{self.dir_suffix}")
        if not exists(self.iteration_dir):
            mkdir(self.iteration_dir)
        fitnesses = self.get_fitnesses(population)
        zipped_population = sorted(zip(fitnesses, population), key=lambda x:x[0])
        # select only top-n candidates from all initial population
        zipped_population = zipped_population[:self.n]
        final_population = [x[1] for x in zipped_population]
        end = time.time()
        log(f"Initialization takes: {end-start:.2f} seconds")
        log(f"Top five cost: {[x[0] for x in zipped_population[:5]]}")
        return final_population

    def random_chromosome(self):
        return [self.random_gene(lim) for lim in self.dim_limit]

    def random_gene(self, n_cells):
        x = random.randint(0, n_cells-1)
        return format(x , f'0{self.bits}b')
    def encode_gene(self, x):
        return format(x , f'0{self.bits}b')
    def decode_gene(self, gene):
        return int(gene, 2)
    def decode_chromosome(self, chromosome):
        chosen_cell_map = {}
        for i, gene in enumerate(chromosome):
            gate_type = self.gate_types[i]
            id = self.decode_gene(gene)
            chosen_cell_map[gate_type] = self.cell_map[gate_type][id]
        return chosen_cell_map
    def fitness(self, chromosome, population_id=0):
        """
        self.iteration dir is assumed to be set 
        before this function is called
        """
        # write genlib
        genlib_path = join(self.iteration_dir, f"{population_id}.genlib")
        chosen_cell_map = self.decode_chromosome(chromosome)
        self.library.write_library_genlib(
            chosen_cell_map, 
            genlib_path,
        )
        # get cost
        cost = self.abcSession.run_ga_genlib(self.design_path, genlib_path)
        return cost  # minimize cost
    def get_fitnesses(self, population, method='wheel'):
        return np.array([self.fitness(chromo, i) for i, chromo in enumerate(population)])
    # def select_parents_rank(self, fitnesses):
    #     parents = [self.population[id] for cost, id in fitnesses[-2:]]
    #     return parents
    def select_parents_rank(self, fitnesses, num_parents=20):
        fitnesses = list(zip(fitnesses, range(self.n)))
        fitnesses.sort()
        top_candidates = fitnesses[:num_parents]
        # Rank weights: [1, 2, 3, ..., num_parents]
        # ranks = np.arange(1, num_parents + 1)
        # Selection probability proportional to ranks (higher rank, higher probability)
        selection_probs = np.array([1/num_parents for _ in range(num_parents)])
        selected_ids = np.random.choice(
            a=[id for cost, id in top_candidates],
            size=2,  # Choose 2 parents
            replace=False,  # Do not allow duplicate selections
            p=selection_probs
        )
        parents = [self.population[id] for id in selected_ids]
        return parents
    def select_parents_wheel(self, fitnesses):
        fitnesses = max(fitnesses) - fitnesses
        total_fitness = sum(fitnesses)
        if total_fitness == 0.0:
            return [self.population[id] for id in [0, 1]]
        probabilities = [f / total_fitness for f in fitnesses]
        choosen_idx = np.random.choice(len(self.population), size=2, p=probabilities)
        parents = [self.population[id] for id in choosen_idx]
        return parents
    # def crossover(self, parent1, parent2):
    #     if random.random() > self.crossover_rate:
    #         return parent1, parent2
    #     child1 = []
    #     child2 = []
    #     for gene1, gene2 in zip(parent1, parent2):
    #         if random.random() > 0.5:
    #             child1.append(gene1)
    #             child2.append(gene2)
    #         else:
    #             child1.append(gene2)
    #             child2.append(gene1)
    #     return child1, child2
    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1, parent2

        point = random.randint(1, self.dim - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def flip(self, gene, i):
        val = '1' if gene[i] == '0' else '0'
        gene = gene[:i] + val + gene[i+1:]
        return gene
    def mutate(self, chromosome):
        if random.random() > self.mutation_rate:
            return chromosome
        for i in range(2):
            gene_idx = random.randint(0, self.dim - 1)
            g = self.random_gene(self.dim_limit[gene_idx])
            chromosome[gene_idx] = g
        return chromosome
    # def mutate(self, chromosome):
    #     if random.random() > self.mutation_rate:
    #         return chromosome

    #     gene_idx = random.randint(0, self.dim - 1)
    #     gene = chromosome[gene_idx]
    #     gate_type = self.gate_types[gene_idx]

    #     mutate_bit_idx = random.randint(0, self.bits - 1)
    #     gene = self.flip(gene, mutate_bit_idx)
    #     # Saturation
    #     if self.decode_gene(gene) >= len(self.cell_map[gate_type]):
    #         gene = self.encode_gene(len(self.cell_map[gate_type]) - 1)
    #     chromosome[gene_idx] = ''.join(gene)
    #     return chromosome

    def evolve(self, iteration):
        new_population = []
        # create dir for current iteration
        self.iteration_dir = join(self.playground_dir, f"{iteration}_{self.dir_suffix}")
        if not exists(self.iteration_dir):
            mkdir(self.iteration_dir)

        # calculate fitnesses for each population
        start = time.time()
        fitnesses = self.get_fitnesses(self.population)
        end = time.time()
        log(f"Fitness calculation takes {end-start:.2f} seconds")
        # for i, c in enumerate(self.population):
        #     for j, g in enumerate(self.population[i]):
        #         print(self.gate_types[j], self.decode_gene(g), end=', ')
            # print(self.decode_chromosome(c))
            # print(f" cost: {fitnesses[i]}")
        best_cost_id = np.argmin(fitnesses)
        cost, chromosome = fitnesses[best_cost_id], self.population[best_cost_id]
        start = time.time()
        for _ in range(self.n // 2):
            parent1, parent2 = self.select_parents_rank(fitnesses, num_parents=self.n//2)
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            new_population.append(self.mutate(child2))
        self.population = new_population
        end = time.time()
        # log(f"Crossover takes {end-start:.2f} seconds")
        return cost, chromosome, best_cost_id


    def run(self):
        ori_mutation_rate = self.mutation_rate
        tar_mutation_rate = 0.01
        self.pq = PriorityQueue()
        seen = {}
        best_cost, best_cost_iteration = float('inf'), -1
        # best_cost_id, best_cost_iteration = -1, -1
        for iteration in range(self.n_iter):
            start = time.time()
            cost, chromosome, best_cost_id = self.evolve(iteration)
            if cost < best_cost:
                best_cost = cost
                best_cost_iteration = iteration
            if ''.join(chromosome) not in seen:
                self.pq.put((-cost, chromosome, best_cost_id, iteration))
                seen[''.join(chromosome)] = True
            if self.pq.qsize() > self.k_solution:
                a = self.pq.get()
                # print(f"poped: {a}")
            # self.mutation_rate -= (ori_mutation_rate - tar_mutation_rate) / self.n_iter
            # print(self.mutation_rate)
            end = time.time()
            log(f"Iteration {iteration}, cost: {cost}, best cost: {best_cost} at {best_cost_iteration}", end='')
            log(f", takes {end-start:.2f} seconds")
        # log(f"Best cost: {best_cost}, at iteration: {best_iteration}")
    def get_results(self):
        min_cell_maps = []
        costs = []
        while not self.pq.empty():
            cost, chromosome, best_cost_id, iteration = self.pq.get()
            costs.append(cost)
            best_cell_id = [self.decode_gene(g) for g in chromosome]
            min_cell_map = {
                g: self.cell_map[g][cell_id] for i, (g, cell_id) in enumerate(zip(self.gate_types, best_cell_id))
            }
            min_cell_maps.append(min_cell_map)
        print(f"top {self.k_solution} cost: {[i for i in costs]}")
        return min_cell_maps, costs