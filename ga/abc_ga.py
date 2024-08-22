from os.path import join
from ga.ga import GA
from DRiLLS.abcSession import abcSession

class AbcGA(GA):
    def __init__(self, design_path, library_path, output_file, actions,
                 choice_commands, n_choice, **kwargs):
        # Initialize the base GA class with any additional arguments
        self.design_path = design_path
        self.library_path = library_path
        self.actions = actions
        self.output_file = output_file
        self.actions_map = {i: action for i, action in enumerate(self.actions)}
        self.choice_commands = choice_commands | {' '}
        self.n_choice = n_choice # suffix length that allow choice command
        kwargs['dir_suffix'] = 'abc_ga' 
        # print(self.choice_commands)
        super(AbcGA, self).__init__(**kwargs)
    def _process_chromosome(self, chromosome):
        """
        Handle choice command, it can only be placed at the end
        of the sequence.
        """
        seq_len = len(chromosome)
        commands = self.decode_chromosome(chromosome)
        # print("before process: ", commands)
        remove = False
        for i in range(seq_len-1, seq_len-self.n_choice-1, -1):
            command = commands[i]
            if command not in self.choice_commands:
                remove = True
            if command in self.choice_commands and remove:
                chromosome[i] = self.random_gene(self.dim_limit[0])
        # for i in range(seq_len-self.n_choice,seq_len-1):
        #     command = commands[i]
        #     next_command = self.decode_gene_to_action(chromosome[i+1])
        #     # if command in self.choice_commands:
        #         # print(command, next_command)
        #     if command in self.choice_commands and next_command not in self.choice_commands:
        #         # print(f"remove {command}, next command: {chromosome[i+1]}")
        #         chromosome[i] = self.random_gene(self.dim_limit[0])
        # print("after process: ", self.decode_chromosome(chromosome))
        return chromosome
    def _choose_share(self):
        """
        choose the command flags for share
        """
        share_large = "st; multi -m -F 150; sop -C 5000000; fx; resyn2"
        share = "share"    # "st; multi -m; sop; fx; resyn2"

    def decode_gene_to_action(self, gene):
        return self.actions_map[self.decode_gene(gene)]
    def decode_chromosome(self, chromosome):
        return [self.actions_map[self.decode_gene(g)] for g in chromosome]
    def _preprocess_population(self):
        self.population = [self._process_chromosome(ch) for ch in self.population]
    def get_fitnesses(self, population, use_hash=True):
        """ 
        Note that we need to save the path for each seen chromosome
        so that in output_netlist, it won't access empty path
        """
        i, end_pos = 0, len(population) - 1
        dests = []
        while i <= end_pos:
            while self.get_chromosome_str(population[i]) in self.seen and end_pos > i:
                population[i], population[end_pos] = population[end_pos], population[i]
                end_pos -= 1
                # print(f"Skip chromosome: {i}")
            if end_pos == 0:
                break
            dests.append(join(self.iteration_dir, f"netlist_{i}.v"))
            i += 1
        costs = []
        command_lists = []
        if end_pos != 0:
            command_lists = [self.decode_chromosome(c) for c in population]
            costs = self.abcSession.run_ga_abc_all(self.design_path, 
                                                self.library_path,
                                                command_lists[:i],
                                                dests)
            # self.seen ==> {chromosome: string: (cost: float, path_to_netlist: str)}
            if use_hash:
                for i in range(len(dests)):
                    self.seen[self.get_chromosome_str(population[i])] = costs[i], dests[i]
                # self.seen.update({self.get_chromosome_str(ch): (cost, dest) for ch, cost, dest in zip(population, costs, dests)})
        if len(costs) < len(population):
            costs.extend(self.seen[self.get_chromosome_str(population[i])][0] for i in range(len(costs), len(population)))
        # [print(i) for i in [self.decode_chromosome(ch) for ch in population]]
        return costs
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
        while not self.pq.empty():
            cost, chromosome, id, iteration = self.pq.get()
            action_seq = self.decode_chromosome(chromosome)
            costs.append(-cost)
            min_action_seqs.append(action_seq)
            if self.pq.empty():
                self.best_iteration = iteration
                self.best_id = id
                print(self.decode_chromosome(chromosome))
                self.best_path = self.seen[self.get_chromosome_str(chromosome)][1]
        print(f"top {self.k_solution} cost: {[i for i in costs]}")
        return min_action_seqs, self.best_iteration, self.best_id, self.best_path, costs
    def output_netlist(self):
        # best_netlist_path = join(self.playground_dir, 
        #             f"{self.best_iteration}_{self.dir_suffix}",
        #             f"netlist_{self.best_id}.v")
        with open(self.best_path, 'r') as src:
            with open(self.output_file, 'w') as dest:
                dest.write(src.read())
        # best_unmapped_netlist_path = best_netlist_path.replace('.v', '_unmapped.v')
        # unmapped_output_file = self.output_file.replace('.v', '_unmapped.v')
        # with open(best_unmapped_netlist_path, 'r') as src:
        #     with open(unmapped_output_file, 'w') as dest:
        #         dest.write(src.read())
            