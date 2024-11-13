import random
import numpy as np
from queue import PriorityQueue
from os import mkdir
from os.path import join, exists
from library import Library
from config import Config
from cost_interface import CostInterface
from abcSession import abcSession
import time
def log(message: str, end='\n'):
    print(message, end=end)
    pass
class GA:
    def __init__(self, n=50, n_init=100, dim=8, dim_limit=[], bits=5, 
                 crossover_rate=0.9, mutation_rate=0.4, mutation_decay=False, n_iter=20, k_solution=5,
                 design_path="", output_path="", session_min_cost=float('inf'), session_start=0, init_population=[],
                 dir_suffix="ga_genlib"):
        self.dir_suffix = dir_suffix
        self.n = n
        self.n_init = n_init
        self.dim = dim
        self.dim_limit = dim_limit
        self.bits = bits
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_decay = mutation_decay
        self.mutate_record = []
        self.n_iter = n_iter
        self.k_solution = k_solution
        self.seen = {}
        self.config = Config()
        if not design_path:
            self.design_path = join(self.config.params['playground_dir'], "design_preprocessed.v")
        else:
            self.design_path = design_path
        self.output_path = output_path
        self.session_min_cost = session_min_cost
        self.session_start = session_start
        self.playground_dir = join(self.config.params['playground_dir'], dir_suffix)
        if not exists(self.playground_dir):
            mkdir(self.playground_dir)
        self.iteration_dir = self.playground_dir
        self.library = Library()
        self.abcSession = abcSession()
        self.cell_map = self.library.cell_map
        self.gate_types = self.library.gate_types
        self.population = self.init_population(init_population)
        print(self.dim_limit)
        # seen chromosomes will be skipped when calculating cost

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
        # if self.dir_suffix == 'ga_genlib':
        #     print(self.design_path)
            # log(f"Top five cost: {[(x[0],[x['cell_name'] for x in self.decode_chromosome(x[1]).values()]) for x in zipped_population[:5]]}")
        log(f"Top five cost: {[x[0] for x in zipped_population[:5]]}")
        return final_population

    def random_chromosome(self):
        return [self.random_gene(lim) for lim in self.dim_limit]

    def random_gene(self, limit):
        if isinstance(limit, int):
            x = random.randint(0, limit-1)
        else:
            x = random.randint(limit[0], limit[1]-1)
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
    def get_chromosome_str(self, chromosome):
        return ','.join(chromosome)
    def fitness(self, chromosome, population_id=0):
        """
        self.iteration dir is assumed to be set 
        before this function is called
        """
        # write genlib
        genlib_path = join(self.iteration_dir, f"{population_id}.genlib")
        dest = genlib_path.replace('.genlib', '.v')
        chosen_cell_map = self.decode_chromosome(chromosome)
        self.library.write_library_genlib(
            chosen_cell_map, 
            genlib_path,
        )
        # get cost
        cost = self.abcSession.run_ga_genlib(self.design_path, genlib_path, dest)
        return cost
    def get_fitnesses(self, population, use_hash=True):
        """
        Get fitness values for whole population.
        If the chromosome is seen before, swap it to 
        the end of the population.
        """
        genlib_paths = []
        # seen_population = [ch for ch in population if self.get_chromosome_str(ch) in self.seen]
        # # Assuming population is a list of strings and self.seen is a set or list of seen elements
        not_seen_pop, seen_pop = [ch for ch in population if self.get_chromosome_str(ch) not in self.seen],  [ch for ch in population if self.get_chromosome_str(ch) in self.seen]
        self.population = not_seen_pop + seen_pop
        for i, ch in enumerate(not_seen_pop):
            genlib_path = join(self.iteration_dir, f"{i}.genlib")
            chosen_cell_map = self.decode_chromosome(ch)
            self.library.write_library_genlib(
                chosen_cell_map, 
                genlib_path,
            )
            genlib_paths.append(genlib_path)
        # while i <= end_pos:
        #     while self.get_chromosome_str(population[i]) in self.seen and end_pos > i:
        #         population[i], population[end_pos] = population[end_pos], population[i]
        #         end_pos -= 1
        #         # print(f"Skip chromosome: {i}")
        #     genlib_path = join(self.iteration_dir, f"{i}.genlib")
        #     chosen_cell_map = self.decode_chromosome(population[i])
        #     # if 'init' in self.iteration_dir:
        #     #     print(f"writing init genlib: {i}.genlib")
        #     self.library.write_library_genlib(
        #         chosen_cell_map, 
        #         genlib_path,
        #     )
        #     genlib_paths.append(genlib_path)
        #     i += 1
        costs = []
        if len(not_seen_pop) != 0:
            dests = [genlib_path.replace('.genlib', '.v') for genlib_path in genlib_paths]
            # print(dests)
            costs = self.abcSession.run_ga_genlib_all(self.design_path, genlib_paths, dests)
            self.seen.update({self.get_chromosome_str(ch): (cost, dest) for ch, cost, dest in zip(not_seen_pop, costs, dests)})
        costs.extend(self.seen[self.get_chromosome_str(ch)][0] for ch in seen_pop)
        # print(len(costs))
        return costs


    # def get_fitnesses(self, population):
    #     fitnesses = []
    #     for i, chromo in enumerate(population): 
    #         dest = join(self.iteration_dir, f"{i}.v")
    #         # print(self.decode_chromosome(chromo))
    #         start = time.time()
    #         fitnesses.append(self.fitness(chromo, i))
    #         end = time.time()
    #         # print(f"{i}-th chromo, cost: {fitnesses[-1]}, takes: {end-start:.2f} sec")
    #         self.seen.update({self.get_chromosome_str(chromo): (fitnesses[-1], dest)})
    #     return np.array(fitnesses)
        # return np.array([self.fitness(chromo, i) for i, chromo in enumerate(population)])
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
        if len(self.mutate_record)==len(self.population):
            self.mutate_record = []
        if random.random() > self.mutation_rate:
            self.mutate_record.append((-1, -1))
            return chromosome
        temp = ()
        for i in range(2):
            gene_idx = random.randint(0, self.dim - 1)
            temp = temp + (gene_idx,)
            g = self.random_gene(self.dim_limit[gene_idx])
            chromosome[gene_idx] = g
        self.mutate_record.append(temp)
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
    def _preprocess_population(self):
        pass
    def evolve(self, iteration):
        new_population = []
        # create dir for current iteration
        self.iteration_dir = join(self.playground_dir, f"{iteration}_{self.dir_suffix}")
        if not exists(self.iteration_dir):
            mkdir(self.iteration_dir)

        # calculate fitnesses for each population
        start = time.time()
        self._preprocess_population()
        # for i, c in enumerate(self.population):
        #     for j, g in enumerate(self.population[i]):
        #         print(self.gate_types[j], self.decode_gene(g), end=', ')
        #     print(f" seen: {self.get_chromosome_str(c) in self.seen}")
        fitnesses = self.get_fitnesses(self.population)
        end = time.time()
        # log(f"Fitness calculation takes {end-start:.2f} seconds")
        # for i, c in enumerate(self.population):
        #     print(self.decode_chromosome(c), "mutate: ", f"cost: {fitnesses[i]}", end='')
        #     if self.mutate_record:
        #         print(f" mutate: {self.mutate_record[i]}")
        #     else:
        #         print()
        # print(f"observed mutation rate: {1 - sum([1 for i in self.mutate_record if i == (-1, -1)])/len(self.population)}")
        # print(f"real mutation rate: {self.mutation_rate}")
        #     for j, g in enumerate(self.population[i]):
        #         print(self.gate_types[j], self.decode_gene(g), end=', ')
        #     # print()
        #     # print(f"chromosome-{i}:",[val['cell_name'] for key, val in self.decode_chromosome(c).items()])
            # print(f" cost: {fitnesses[i]},  seen: {self.get_chromosome_str(c) in self.seen}")
        best_cost_id = np.argmin(fitnesses)
        # chromosome act like pointer here, takes 2 hours to find. fuck
        cost, chromosome = fitnesses[best_cost_id], self.population[best_cost_id][:]
        # print(f"Best cost: {cost}")
        # print(f"Best chromosome: {self.decode_chromosome(chromosome)}")
        start = time.time()
        for _ in range(self.n // 2):
            parent1, parent2 = self.select_parents_rank(fitnesses, num_parents=self.n//2)
            child1, child2 = self.crossover(parent1, parent2)
            mchild1 = self.mutate(child1[:])
            mchild2 = self.mutate(child2[:])
            # for a, b in zip(child1, mchild1):
            #     if a != b:
            #         if self.dir_suffix == 'abc_ga':
            #             print(f"Ori: {self.decode_gene_to_action(a)}, mutated: {self.decode_gene_to_action(b)}")
            new_population.append(mchild1)
            new_population.append(mchild2)
        self.population = new_population
        end = time.time()
        # log(f"Crossover takes {end-start:.2f} seconds")
        # print(f"Best chromosome: {self.decode_chromosome(chromosome)}")
        # print(f"Best chromosome: {chromosome}")
        return cost, chromosome, best_cost_id


    def run(self):
        ori_mutation_rate = self.mutation_rate
        tar_mutation_rate = 0.1
        self.pq = PriorityQueue()
        seen = {}
        best_cost, best_cost_iteration = float('inf'), -1
        # best_cost_id, best_cost_iteration = -1, -1
        evolve_time = 0
        for iteration in range(self.n_iter):
            start = time.time()
            cost, chromosome, best_cost_id = self.evolve(iteration)
            end = time.time()
            evolve_time += end-start
            # if best, save netlist to output path
            if cost < best_cost:
                best_cost = cost
                best_cost_iteration = iteration
            if cost < self.session_min_cost:
                self.session_min_cost = cost
                self.save_netlist(iteration, best_cost_id, chromosome)
            if ''.join(chromosome) not in seen:
                self.pq.put((-cost, chromosome, best_cost_id, iteration))
                seen[''.join(chromosome)] = True
            if self.pq.qsize() > self.k_solution:
                a = self.pq.get()
                # print(f"poped: {a}")
            if self.mutation_decay:
                self.mutation_rate -= (ori_mutation_rate - tar_mutation_rate) / self.n_iter
            # print(self.mutation_rate)
            end = time.time()
            log(f"Iteration {iteration}, cost: {cost}, best cost: {best_cost} at {best_cost_iteration}", end='')
            log(f", takes {end-start:.2f} seconds")
            print(f"remaining time: {10800-(end-self.session_start)}")
            # remaining time smaller than 8 minutes
            if 10800 - (end - self.session_start) < 480 + end-start:
                print(f"Terminate GA at iteration: {iteration}, current time: {end}, start time: {self.session_start}, time elapsed: {end - self.session_start}")
                break
        log(f"Evolve total takes: {evolve_time:.2f}")
        # log(f"Best cost: {best_cost}, at iteration: {best_iteration}")
    def save_netlist(self, iteration, id, chromosome):
        # src = join(self.iteration_dir, f"{id}.v")
        src = self.seen[self.get_chromosome_str(chromosome)][1]
        with open(src, 'r') as best_netlist:
            with open(self.output_path, 'w') as output_path:
                output_path.write(best_netlist.read())
        print(f"{src} is written to {self.output_path}")


    def get_results(self):
        min_cell_maps = []
        costs = []
        while not self.pq.empty():
            cost, chromosome, best_cost_id, iteration = self.pq.get()
            costs.append(-cost)
            best_cell_id = [self.decode_gene(g) for g in chromosome]
            min_cell_map = {
                g: self.cell_map[g][cell_id] for i, (g, cell_id) in enumerate(zip(self.gate_types, best_cell_id))
            }
            min_cell_maps.append(min_cell_map)
            if self.pq.empty():
                with open("best_ga.txt", 'w') as f:
                    f.write(' '.join([x['cell_name'] for x in self.decode_chromosome(chromosome).values()]))
        print(f"top {self.k_solution} cost: {[i for i in costs]}")
        return min_cell_maps, costs