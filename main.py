import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import datetime
import numpy as np
import time
import yaml
from parseArgs import *
# from DRiLLS.model import A2C
from DRiLLS.abcSession import abcSession
from utils.cost_interface import CostInterface
from utils.library import Library
from utils.config import Config
from utils.post_optimizor import PostOptimizor
from sa.sa import SA
from ga.ga import GA
from ga.abc_ga import AbcGA
# from drills.fixed_optimization import optimize_with_fixed_script
# from pyfiglet import Figlet

def log(message):
    print('[DRiLLS {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + "] " + message)
    pass

if __name__ == '__main__':
    # read args and yaml
    args = parse_arguments()
    with open('params.yml', 'r') as file:
        params = yaml.safe_load(file)
    params['library_file'] = args.library
    params['design_file'] = args.netlist
    params['cost_estimator'] = args.cost_function
    params['output_file'] = args.output
    print(params)
    # init modules
    config = Config(
        params=params
    )
    cost_interface = CostInterface(
        cost_estimator_path=args.cost_function,
        library_path=args.library                           
    )
    library = Library(
        library_path=args.library                           
    )
    abc_session = abcSession()
    
    #phase 1 GA
    ga = GA(
        n=params['ga_n'],
        n_init=params['ga_n_init'],
        n_iter=params['ga_n_iter'],
        init_population=[[x['id'] for x in library.min_cell_map.values()]],
        dim_limit=[len(library.cell_map[gate_type]) for gate_type in library.gate_types],
        k_solution=5
    )
    ga.run()
    best_cell_maps, costs = ga.get_results()
    min_cost = float('inf')
    min_cell_map = None
    # write best genlibs
    best_cells_map = {gate_type: [] for gate_type in library.gate_types}
    for i, best_cell_map in enumerate(best_cell_maps):
        for gate_type, cell in best_cell_map.items():
            if cell not in best_cells_map[gate_type]:
                best_cells_map[gate_type].append(cell)
        library.write_library_genlib(
            best_cell_map, 
            f"./lib/library_{i}.genlib",
        )
        if i == len(best_cell_maps) - 1:
            library.write_library_genlib(
                best_cell_map, 
                f"./lib/library.genlib",
            )
            min_cost = -costs[i]
            min_cell_map = best_cell_map
    library.write_library_genlib_all(
        dest="./lib/library_multi.genlib",
        cell_map=best_cells_map
    )
    # for gate, cells in best_cells_map.items():
    #     print(f"{gate}: ", end='')
    #     for cell in cells:
    #         print(cell['cell_name'], end='')
    sa_init_solution = []
    for i, (gate_type, cells) in enumerate(best_cells_map.items()):
        for j, cell in enumerate(cells):
            sa_init_solution.append(cell['cost'])
    print(len(sa_init_solution))
    sa = SA(
        init_solution=sa_init_solution, 
        init_cost=min_cost, 
        cell_map=best_cells_map,
        temp_h=4000000, temp_l=0.1, 
        cooling_rate=0.8, num_iterations=1000,
    )
    min_cell_map, min_cost = sa.run()
    library.write_library_genlib_all(
        cell_map=min_cell_map, 
        dest=f"./lib/library.genlib",
    )
    # phase 2 abc_ga
    abc_ga = AbcGA(
        design_path=abc_session.preprocessed_design_path,
        output_file=params['output_file'],
        library_path='./lib/library.genlib',
        actions=params['actions'],
        mutation_rate=0.5,
        n=params['abc_ga_n'],
        n_init=params['abc_ga_n_init'],
        n_iter=params['abc_ga_n_iter'],
        dim_limit=[len(params['actions']) for _ in range(params['abc_ga_seq_len'])],
    )
    print([len(params['actions']) for _ in range(params['abc_ga_seq_len'])])
    abc_ga.run()
    best_action_seqs, best_iteration, best_id, costs = abc_ga.get_results()
    abc_ga.output_netlist()
    # write best netlists
    min_cost = min(costs)
    
    opt = PostOptimizor()
    # phase 3 post map
    best_netlist, cost = opt.post_map(params['output_file'].replace('.v', '_unmapped.v'))
    if cost < min_cost:
        with open(best_netlist, 'r') as src:
            with open(params['output_file'], 'w') as dest:
                dest.write(src.read())
    # phase 4 add buffer
    buffer_temp_dir = os.path.join(params['playground_dir'], "buffer")
    if not os.path.exists(buffer_temp_dir):
        os.mkdir(buffer_temp_dir)
    min_idx = -1
    best_buffer_netlist = ""
    # find best buffer to insert
    start = time.time()
    for buf_cell in library.cell_map['buf']:
        buf_cell_name = buf_cell['cell_name']
        for i, max_fanout in enumerate(range(2, 22, 2)):
            file_name = params['output_file'].replace('.v', f"_{buf_cell_name}_fan_{max_fanout}.v")
            dest = os.path.join(buffer_temp_dir, file_name)
            opt.insert_buffers(params['output_file'], dest_path=dest, max_fanout=max_fanout, buf_cell_name=buf_cell_name)
            cost = cost_interface.get_cost(dest)
            print(f"max fanout: {max_fanout}, cost: {cost}")
            if cost < min_cost:
                min_cost = cost
                min_idx = i
                best_buffer_netlist = dest
    if best_buffer_netlist:
        print("buf: ", best_buffer_netlist)
        with open(best_buffer_netlist, 'r') as src:
            with open(params['output_file'], 'w') as dest:
                dest.write(src.read())
    end = time.time()
    print(end - start)
    # phase 5 gate sizing
    best_netlist, cost = opt.run_gate_sizing(params['output_file'])
    print(f"gate sizing: {best_netlist}, {cost}")
    with open(best_netlist, 'r') as src:
        with open(params['output_file'], 'w') as dest:
            dest.write(src.read())
    print(f"FINAL_min_cost: {cost_interface.get_cost(params['output_file'])}")
    
