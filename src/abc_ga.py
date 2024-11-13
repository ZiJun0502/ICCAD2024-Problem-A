from os.path import join
from .ga import GA
from .abcSession import abcSession

class AbcGA(GA):
    def __init__(self, design_path, library_path, 
                 actions, seq_len,
                 choice_commands, n_choice, **kwargs):
        self.abcSession = abcSession()
        # Initialize the base GA class with any additional arguments
        self.design_path = design_path
        self.library_path = library_path
        self.actions = actions

        design_path = self._filter_commands()
        kwargs['design_path'] = design_path
        self.seq_len = seq_len
        kwargs['dim'] = self.seq_len
        self.actions_map = {i: action for i, action in enumerate(self.actions)}
        # choice commands
        self.choice_commands = choice_commands | {' '}
        len_choice_commands = 4
        num_command_types = len(self.actions)
        len_choices = sum(i in choice_commands for i in self.actions)
        kwargs['dim_limit'] = [(0, num_command_types) for _ in range(self.seq_len)]
        kwargs['dim_limit'] = [(0, num_command_types-len_choices) for _ in range(self.seq_len-len_choice_commands)] + \
                    [(num_command_types-len_choices-2, num_command_types) for _ in range(len_choice_commands)]
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
        return chromosome
    def _filter_commands(self):
        """
        choose the command flags for share
        """
        # get cost of original netlist
        command = "st;"
        dest = self.design_path.replace('.v', '_baseline_mapped.v')
        self.abcSession.forward_command(command=command, src = self.design_path, dest=dest, library_path=self.library_path, mapping=True)
        baseline_cost = self.abcSession.cost_interface.get_cost(dest)

        # run satclp
        command = 'satclp -C 200; fx; st'
        satclp_dest = self.design_path.replace('.v', '_satclp_mapped.v')
        satclp_dest_unmapped = self.design_path.replace('.v', '_satclp_unmapped.v')
        abc_output = self.abcSession.forward_command(command=command, src=self.design_path, dest=satclp_dest, library_path=self.library_path, mapping=True)
        abc_output = abc_output.decode().split('\n')
        try:
            cost = self.abcSession.cost_interface.get_cost(satclp_dest)
        except:
            cost = float('inf')
        print(f"satclp cost: {cost}, original cost: {baseline_cost}")

        error_abc = any(line.startswith("Error") for line in abc_output)
        if error_abc or cost >= baseline_cost:
            pass
            # id_rm = self.actions.index(command)
            # self.actions = self.actions[:id_rm] + self.actions[id_rm+1:]
        else:
            self.abcSession.unmap(satclp_dest, satclp_dest_unmapped, self.library_path)
            self.design_path = satclp_dest_unmapped
            # with open(dest, 'r') as satclp_netlist:
            #     with open(self.design_path, 'w') as ori_netlist:
            #         ori_netlist.write(satclp_netlist.read()) 
        # print(self.actions)
        return self.design_path
        # for line in abc_output:
        #     print(line)
        #     print()




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
        not_seen_pop, seen_pop = [ch for ch in population if self.get_chromosome_str(ch) not in self.seen],  [ch for ch in population if self.get_chromosome_str(ch) in self.seen]
        self.population = not_seen_pop + seen_pop
        dests = [join(self.iteration_dir, f"netlist_{i}.v") for i in range(len(not_seen_pop))]
        # while i <= end_pos:
        #     while self.get_chromosome_str(population[i]) in self.seen and end_pos > i:
        #         population[i], population[end_pos] = population[end_pos], population[i]
        #         end_pos -= 1
        #         # print(f"Skip chromosome: {i}")
        #     if end_pos == 0:
        #         break
        #     dests.append(join(self.iteration_dir, f"netlist_{i}.v"))
        #     i += 1
        costs = []
        command_lists = []
        if len(not_seen_pop) != 0:
            command_lists = [self.decode_chromosome(c) for c in not_seen_pop]
            costs = self.abcSession.run_ga_abc_all(self.design_path, 
                                                self.library_path,
                                                command_lists,
                                                dests)
            # self.seen ==> {chromosome: string: (cost: float, path_to_netlist: str)}
            # if use_hash:
                # for i in range(len(dests)):
                #     self.seen[self.get_chromosome_str(population[i])] = costs[i], dests[i]
            self.seen.update({self.get_chromosome_str(ch): (cost, dest) for ch, cost, dest in zip(not_seen_pop, costs, dests)})
        costs.extend(self.seen[self.get_chromosome_str(ch)][0] for ch in seen_pop)
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
                with open("abc_ga_best.txt", 'w') as f:
                    f.write('; '.join(self.decode_chromosome(chromosome)))
                self.best_path = self.seen[self.get_chromosome_str(chromosome)][1]
        print(f"top {self.k_solution} cost: {[i for i in costs]}")
        return min_action_seqs, self.best_iteration, self.best_id, self.best_path, costs
    def output_netlist(self, output_path):
        # best_netlist_path = join(self.playground_dir, 
        #             f"{self.best_iteration}_{self.dir_suffix}",
        #             f"netlist_{self.best_id}.v")
        with open(self.best_path, 'r') as src:
            with open(output_path, 'w') as dest:
                dest.write(src.read())
        # best_unmapped_netlist_path = best_netlist_path.replace('.v', '_unmapped.v')
        # unmapped_output_file = self.output_path.replace('.v', '_unmapped.v')
        # with open(best_unmapped_netlist_path, 'r') as src:
        #     with open(unmapped_output_file, 'w') as dest:
        #         dest.write(src.read())
            