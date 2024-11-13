import numpy as np
import time
from os.path import join, exists 
from os import mkdir
from .cost_interface import CostInterface
from .library import Library
from .config import Config
from .abcSession import abcSession
class SA:
    def __init__(self, init_solution, init_cost, cell_map,
                 temp_h=1000, temp_l = 0.1, cooling_rate=0.95, num_iterations=100,
                 dir_suffix="sa", design_path=""):
        """
        Initializes the Simulated Annealing algorithm.
        
        Parameters:
        - cost_function: Function to evaluate the cost (objective function).
        - initial_solution: Initial guess for the solution (array of 8 continuous variables).
        - temperature: Initial temperature for the annealing process.
        - cooling_rate: Rate at which the temperature decreases.
        - num_iterations: Number of iterations to perform.
        - step_size: Maximum step size for generating new solutions.
        """
        self.cost_interface = CostInterface()
        self.library = Library()
        self.abcSession = abcSession()
        self.config = Config()
        self.playground_dir = join(self.config.params['playground_dir'], dir_suffix)
        if not exists(self.playground_dir):
            mkdir(self.playground_dir)
        self.init_solution = np.array(init_solution)
        self.current_solution = np.array(init_solution)
        self.temp_h = temp_h
        self.temp_l = temp_l
        self.temperature = temp_h
        self.cooling_rate = cooling_rate
        self.num_iterations = num_iterations
        self.cell_map = cell_map
        self.design_path = design_path if design_path else join(self.config.params['playground_dir'], "design_preprocessed.v")

        self.print_freq = 50
        self.best_solution = np.copy(self.current_solution)
        self.best_cost = init_cost
        self.best_iteration = -1
        
    def _generate_neighbor(self):
        """
        Generates a new neighboring solution by perturbing the current solution.
        Each variable has a different step size within the range (-step_size[i], step_size[i]).
        """
        neighbor = np.copy(self.current_solution)
        for i in range(len(neighbor)):
            perturbation = np.random.normal(0, 0.006 * neighbor[i])
            neighbor[i] += perturbation
        # perturbation = np.random.normal(0, 0.05)
        # neighbor += perturbation
        
        # Ensure the values are within the desired range [0, 100]
        neighbor = np.clip(neighbor, 0, 100)
        return neighbor

    
    def _acceptance_probability(self, current_cost, neighbor_cost):
        """
        Computes the acceptance probability of a neighbor solution based on the current temperature.
        """
        if neighbor_cost < current_cost:
            return 1.0
        else:
            return np.exp((current_cost - neighbor_cost) / self.temperature)
    
    def run(self):
        """
        Runs the Simulated Annealing algorithm to find the minimum cost solution.
        """
        current_cost = self.get_sol_cost(self.current_solution, 0)
        start, end = time.time(), time.time()
        for iteration in range(self.num_iterations):
            neighbor_solution = self._generate_neighbor()
            neighbor_cost = self.get_sol_cost(neighbor_solution, iteration)
            if iteration % self.print_freq == 0:
                end = time.time()
                # print(f"Temp: {self.temperature}")
                # print([f'{x:.4f}' for x in neighbor_solution])
                print(f"Iteration {iteration}, cost: {neighbor_cost}, best cost: {self.best_cost} at {self.best_iteration}, time elapsed: {end-start:.2f}")
                start = time.time()
            # Accept the neighbor solution with certain probability
            if np.random.rand() < self._acceptance_probability(current_cost, neighbor_cost):
                self.current_solution = neighbor_solution
                if neighbor_cost < self.best_cost:
                    self.best_solution = neighbor_solution
                    self.best_cost = neighbor_cost
                    self.best_iteration = iteration
            
            # Cool down the temperature
            self.temperature *= self.cooling_rate
            # self.temperature = self.temp_l + (self.temp_h-self.temp_l) * (self.num_iterations-iteration) / self.num_iterations
            self.temperature = max(self.temp_l, self.temperature)
        
        self.set_cell_map(self.best_solution)
        return self.cell_map, self.best_cost
    def get_sol_cost(self, solution, iteration):
        self.set_cell_map(solution)
        library_path = join(self.playground_dir, f"{iteration}.genlib")
        target_netlist_path = join(self.playground_dir, f"{iteration}_mapped.v")
        self.library.write_library_genlib_all(
            cell_map=self.cell_map,
            dest=library_path,
        )
        self.abcSession.run_ga_genlib(
            design_path=self.design_path,
            library_path=library_path,
            dest=target_netlist_path
        )
        return self.cost_interface.get_cost(target_netlist_path)
    # def set_cell_map(self, solution):
    #     for i, (gate_type, cell) in enumerate(self.cell_map.items()):
    #         self.cell_map[gate_type]['cost'] = solution[i]
    def set_cell_map(self, solution):
        idx = 0
        for i, (gate_type, cells) in enumerate(self.cell_map.items()):
            for j, cell in enumerate(cells):
                cell['cost'] = solution[idx]
                idx+=1
                self.cell_map[gate_type][j] = cell
