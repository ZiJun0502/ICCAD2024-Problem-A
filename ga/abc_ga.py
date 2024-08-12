from os.path import join
from ga.ga import GA
from DRiLLS.abcSession import abcSession

class AbcGA(GA):
    def __init__(self, design_path, library_path, output_file, actions, **kwargs):
        # Initialize the base GA class with any additional arguments
        self.design_path = design_path
        self.library_path = library_path
        self.actions = actions
        self.actions_map = {i: action for i, action in enumerate(self.actions)}
        self.output_file = output_file
        kwargs['dir_suffix'] = 'abc_ga' 
        super(AbcGA, self).__init__(**kwargs)
    def decode_chromosome(self, chromosome):
        return [self.actions_map[self.decode_gene(g)] for g in chromosome]
    def fitness(self, chromosome, population_id=0):
        actions = self.decode_chromosome(chromosome)
        dest = join(self.iteration_dir, f"netlist_{population_id}.v")
        cost = self.abcSession.run_ga_abc(self.design_path, 
                                          self.library_path, 
                                          actions, 
                                          dest=dest)
        return cost
    def get_results(self):
        min_action_seqs = []
        costs = []
        best_iteration, best_id = -1, -1
        while not self.pq.empty():
            cost, chromosome, id, iteration = self.pq.get()
            action_seq = self.decode_chromosome(chromosome)
            costs.append(cost)
            min_action_seqs.append(action_seq)
            if self.pq.empty():
                self.best_iteration = iteration
                self.best_id = id
        print(f"top {self.k_solution} cost: {[i for i in costs]}")
        return min_action_seqs, best_iteration, best_id, costs
    def output_netlist(self):
        best_netlist_path = join(self.playground_dir, 
                    f"{self.best_iteration}_{self.dir_suffix}",
                    f"netlist_{self.best_id}.v")
        best_unmapped_netlist_path = best_netlist_path.replace('.v', '_unmapped.v')
        unmapped_output_file = self.output_file.replace('.v', '_unmapped.v')
        with open(best_netlist_path, 'r') as src:
            with open(self.output_file, 'w') as dest:
                dest.write(src.read())
        with open(best_unmapped_netlist_path, 'r') as src:
            with open(unmapped_output_file, 'w') as dest:
                dest.write(src.read())
            