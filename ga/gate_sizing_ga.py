import random
import numpy as np
from queue import PriorityQueue
from os import mkdir
from os.path import join, exists
from utils.library import Library
from utils.config import Config
from utils.cost_interface import CostInterface
from DRiLLS.abcSession import abcSession
from bisect import bisect_right, bisect_left
import time
import copy

class GateSizingGA:
    def __init__(self, n=50, n_init=100, dim=8, id_dim_limit=[], bits=5, 
                 crossover_rate=0.9, mutation_rate=0.4, n_iter=20,
                 design_path="", output_path="", session_min_cost=float('inf'), init_population=[],
                 dir_suffix="gate_sizing_ga"):
        self.config = Config()
        self.library = Library()
        self.abcSession = abcSession()
        self.cost_interface = CostInterface()
        self.dir_suffix = dir_suffix
        self.design_path = design_path
        with open(self.design_path, 'r') as f:
            self.design_str = f.readlines()
        self.output_path = output_path
        self.session_min_cost = session_min_cost
        self.playground_dir = join(self.config.params['playground_dir'], dir_suffix)
        if not exists(self.playground_dir):
            mkdir(self.playground_dir)
        self.iteration_dir = self.playground_dir
        self.cell_list = self.preprocess_format_cell_map(self.library.cell_map)
        self.id_to_cell_name = self.get_id_cell_map(self.library.cell_map)
        self.gate_types = self.library.gate_types

        self.n = n
        self.n_init = n_init
        self.dim = dim
        self.id_dim_limit = id_dim_limit
        self.gate_pos_list, self.pos_dim_limit = self.get_pos_dim_limit(self.design_path)
        self.bits = bits
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_iter = n_iter
        self.seen = {}
        self.population = self.init_population(init_population)
        # seen chromosomes will be skipped when calculating cost
    def get_id_cell_map(self, cell_map):
        id_to_cell_name = {gate: [] for gate in cell_map.keys()}
        for gate, cells in cell_map.items():
            id_to_cell_name[gate] = [cell['cell_name'] for cell in cells]
        # print("id to cellname", id_to_cell_name)
        return id_to_cell_name
    def get_pos_dim_limit(self, design_file):
        gate_counts = {gate: 0 for gate in self.gate_types}
        gate_pos_list = {gate: [] for gate in self.gate_types}

        # Open the file and iterate through its lines
        with open(design_file, 'r') as file:
            for i, line in enumerate(file):
                # Strip whitespace from the beginning and end of the line
                line = line.strip()

                # Check for each gate type prefix using startswith
                for gate_type in self.gate_types:
                    if line.startswith(gate_type):
                        gate_pos_list[gate_type].append(i)
                        break  # Exit the loop after finding the match
        # print(gate_counts)
        pos_dim_limit = [len(pos) for pos in gate_pos_list.values()]
        # print(pos_dim_limit)
        # print(gate_pos_list)
        return gate_pos_list, pos_dim_limit
        
        # Convert defaultdict to a regular dict and return
        return dict(gate_counts)

    def preprocess_format_cell_map(self, cell_map):
        result = []
        for gate_type, cells in cell_map.items():
            pairs = [(cell['cell_name'], cell['id'], cell['cost']) for cell in cells]
            pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
            result.append([(x[0], x[1]) for x in pairs_sorted])
        # print(result)
        return result

    def init_population(self, init_population):
        """
        Build a random population of size n_init, 
        and only preserve top-n candidates.
        """
        start = time.time()
        population = []
        # print(init_population)
        population = [self.random_chromosome() for _ in range(self.n_init-len(init_population))]
        # for i, c in enumerate(population):
        #     print(f"chromosome {i:2d}: {c}")
        # seeded initialization
        if len(init_population):
            for cell_map in init_population:
                # for gate_type, cell in cell_map.items():
                init_chromosome = [[(0, self.pos_dim_limit[i], cell['id'])] for i, cell in enumerate(cell_map.values())]
                population += [init_chromosome]
                # print(f"seeded chromosome: {init_chromosome}")
        else:
            cell_map = {gate: '' for gate in self.gate_types}
            for line in self.design_str:
                line = line.strip()
                for gate in cell_map.keys():
                    if line.strip().startswith(gate):
                        cell_map[gate] = line.split()[0]
            cell_map_ = []
            for i, (gate_type, cell_name) in enumerate(cell_map.items()):
                fid = 0
                for id in range(self.id_dim_limit[i]):
                    if self.id_to_cell_name[gate_type][id] == cell_name:
                        fid = id
                        break
                cell_map_.append(fid)
            init_chromosome = [[(0, self.pos_dim_limit[i], id)] for i, id in enumerate(cell_map_)]
            population += [init_chromosome]
        self.iteration_dir = join(self.playground_dir, f"init_{self.dir_suffix}")
        if not exists(self.iteration_dir):
            mkdir(self.iteration_dir)
        # print(f"init population:\n {population}")
        fitnesses = self.get_fitnesses(population)
        # print(f"get_fit init population:\n {population}")
        zipped_population = sorted(zip(fitnesses, population), key=lambda x:x[0])
        # select only top-n candidates from all initial population
        zipped_population = zipped_population[:self.n]
        final_population = [x[1] for x in zipped_population]
        end = time.time()
        return final_population

    def random_chromosome(self):
        # select random cell, position: 0~ 
        # return [[(0, g=self.random_gene(id_lim)), (pos_lim, g)] for id_lim, pos_lim in zip(self.id_dim_limit, self.pos_dim_limit)]
        return [[(0, pos_lim, g)] for id_lim, pos_lim in zip(self.id_dim_limit, self.pos_dim_limit) for g in [self.random_gene(id_lim)]]


    def random_gene(self, limit):
        if isinstance(limit, int):
            x = random.randint(0, limit-1)
        else:
            x = random.randint(limit[0], limit[1]-1)
        return x
    # def encode_gene(self, x):
    #     return format(x , f'0{self.bits}b')
    # def decode_gene(self, gene):
    #     return int(gene, 2)
    def decode_chromosome(self, chromosome):
        chosen_cell_map = {}
        for i, gene in enumerate(chromosome):
            gate_type = self.gate_types[i]
            id = self.decode_gene(gene)
            chosen_cell_map[gate_type] = self.cell_map[gate_type][id]
        return chosen_cell_map
    # def get_chromosome_str(self, chromosome):
    #     print(chromosome)
    #     return ','.join(chromosome)
    def replace_cell_line(self, line, cell_name):
        parts = line.strip().split()
        current_cell_name = parts[0]
        new_line = line.replace(current_cell_name, cell_name, 1)
        return new_line
    def write_gate_sizing_file(self, chromosome, dest):
        lines = self.design_str[:]
        for i, (gate, li) in enumerate(zip(self.gate_types, chromosome)):
            # li: list[tuple(posistion, cell_id)]
            # print(gate, li)
            for (start, end, cell_id) in li:
                # encounter last
                line_to_write = self.gate_pos_list[gate][start:end]
                # print(line_to_write)
                cell_name = self.id_to_cell_name[gate][cell_id]
                for line_id in line_to_write:
                    ori_line = lines[line_id] 
                    modified_line = self.replace_cell_line(ori_line, cell_name)
                    # print("ori:", ori_line)
                    # print("mod:", modified_line)
                    lines[line_id] = modified_line
        # print(f"dest: {dest}")
        with open(dest, 'w') as f:
            f.write(''.join(lines))
        with open(dest.replace('.v', '.txt'), 'w') as f:
            data = self.encode_chromosome(chromosome)
            for i, outer_tuple in enumerate(data):
                f.write(f"{self.gate_types[i]} ")
                for inner_tuple in outer_tuple:
                    # Convert the inner tuple to a string and write it to the file
                    f.write(f"{inner_tuple}, ")
                f.write('\n')
    def fitness(self, chromosome, population_id=0):
        """
        self.iteration dir is assumed to be set 
        before this function is called
        """
        dest = join(self.iteration_dir, f"{population_id}.v")
        # replace cell with chromosome posistion:id encoding
        self.write_gate_sizing_file(chromosome, dest)
        cost = self.cost_interface.get_cost(dest)
        return cost


    def get_fitnesses(self, population):
        fitnesses = []
        for i, chromo in enumerate(population): 
            dest = join(self.iteration_dir, f"{i}.v")
            start = time.time()
            fitnesses.append(self.fitness(chromo, i))
            end = time.time()
            # self.seen.update({self.get_chromosome_str(chromo): (fitnesses[-1], dest)})
        return np.array(fitnesses)
        # return np.array([self.fitness(chromo, i) for i, chromo in enumerate(population)])
    # def select_parents_rank(self, fitnesses):
    #     parents = [self.population[id] for cost, id in fitnesses[-2:]]
    #     return parents
    def select_parents_rank(self, fitnesses, num_parents=30):
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

    def crossover(self, parent1, parent2):
        def crossover_for_gate_type(p1, p2, pos_limit):
            # Find the crossover point
            if pos_limit < 2:
                return p1, p2
            
            point = random.randint(1, pos_limit - 1)
            # print(f"Gate type: {self.gate_types[i]}, len: {pos_limit}, selected point: {point}")

            # Use binary search to find the split point
            index1 = bisect_right([st for st, end, _ in p1], point)
            index2 = bisect_right([st for st, end, _ in p2], point)
            p1_left  = copy.deepcopy(p1[:index1])
            p1_right = copy.deepcopy(p1[index1-1:])
            p2_left  = copy.deepcopy(p2[:index2])
            p2_right = copy.deepcopy(p2[index2-1:])
            # print(len(p1), index1)
            p1_mid   = p1[index1-1][:]
            p2_mid   = p2[index2-1][:]
            if p1_left[-1][0] == point:
                p1_left.pop()
            else:
                p1_left[-1] = (p1_left[-1][0], point, p1_left[-1][2])
            if p2_left[-1][0] == point:
                p2_left.pop()
            else:
                p2_left[-1] = (p2_left[-1][0], point, p2_left[-1][2])
            if p1_right[0][1] == point:
                p1_right.pop()
            else:
                p1_right[0] = (point, p1_right[0][1], p1_right[0][2])
            if p2_right[0][1] == point:
                p2_right.pop()
            else:
                p2_right[0] = (point, p2_right[0][1], p2_right[0][2])
            # print(f"p1_left, {p1_left}")
            # print(f"p1_mid, {p1_mid}")
            # print(f"p1_right, {p1_right}")
            # print(f"p2_left, {p2_left}")
            # print(f"p2_mid, {p2_mid}")
            # print(f"p2_right, {p2_right}")
            if p2[index2-1][0] != point:
                child1 = p1_left + p2_right
            else:
                child1 = p1_left + p2_right
            if p1[index1-1][0] != point:
                child2 = p2_left + p1_right
            else:
                child2 = p2_left + p1_right
            # child2 = p2[:index] + p1[index1:]
            
            return child1, child2
        # single point
        if random.random() > self.crossover_rate:
            return parent1, parent2
        child1 = []
        child2 = []

        # Perform crossover for each gate type
        for i, (p1, p2) in enumerate(zip(parent1, parent2)):
            # print("p1", p1)
            # print("p2", p2)
            gate1, gate2 = crossover_for_gate_type(p1, p2, self.pos_dim_limit[i])
            # print(f"gate1: {gate1}")
            # print(f"gate2: {gate2}")
            child1.append(gate1)
            child2.append(gate2)

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

            for j in range(2):
                base_idx = random.randint(0, len(chromosome[gene_idx])-1)
                tup = chromosome[gene_idx][base_idx]
                chromosome[gene_idx][base_idx] = (tup[0], tup[1], random.randint(0, self.id_dim_limit[gene_idx]-1))
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
        fitnesses = self.get_fitnesses(self.population)
        end = time.time()
        # log(f"Fitness calculation takes {end-start:.2f} seconds")
        # for i, c in enumerate(self.population):
        #     print(f"chromosome-{i}: {c} cost: {fitnesses[i]}")
        best_cost_id = np.argmin(fitnesses)
        # chromosome act like pointer here, takes 2 hours to find. fuck
        cost, chromosome = fitnesses[best_cost_id], self.population[best_cost_id][:]
        # print(f"Best cost: {cost}")
        # print(f"Best chromosome: {self.decode_chromosome(chromosome)}")
        start = time.time()
        for _ in range(self.n // 2):
            parent1, parent2 = self.select_parents_rank(fitnesses, num_parents=self.n//2)
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            new_population.append(self.mutate(child2))
            # new_population.append(child1)
            # new_population.append(child2)
        self.population = new_population
        end = time.time()
        # log(f"Crossover takes {end-start:.2f} seconds")
        # print(f"Best chromosome: {self.decode_chromosome(chromosome)}")
        # print(f"Best chromosome: {chromosome}")
        return cost, chromosome, best_cost_id

    def encode_chromosome(self, chromosome):
        return tuple(tuple(inner_list) for inner_list in chromosome)
    def run(self):
        ori_mutation_rate = self.mutation_rate
        tar_mutation_rate = 0.1
        self.pq = PriorityQueue()
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
                dest = join(self.iteration_dir, f"{best_cost_id}.v")
                self.best_path = dest
            if cost < self.session_min_cost:
                self.session_min_cost = cost
                # self.save_netlist(iteration, best_cost_id, chromosome)
                # print(f"poped: {a}")
            self.mutation_rate -= (ori_mutation_rate - tar_mutation_rate) / self.n_iter
            # print(self.mutation_rate)
            end = time.time()
            print(f"Iteration {iteration}, cost: {cost}, best cost: {best_cost} at {best_cost_iteration}", end='')
            print(f", takes {end-start:.2f} seconds")
        print(f"Evolve total takes: {evolve_time:.2f}")
        print(self.best_path)
        return self.best_path, best_cost
    def save_netlist(self, iteration, id, chromosome):
        # src = join(self.iteration_dir, f"{id}.v")
        # src = self.seen[self.get_chromosome_str(chromosome)][1]
        # with open(src, 'r') as best_netlist:
        #     with open(self.output_path, 'w') as output_path:
        #         output_path.write(best_netlist.read())
        # print(f"{src} is written to {self.output_path}")
        pass


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