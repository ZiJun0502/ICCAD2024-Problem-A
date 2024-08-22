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

import cProfile
import pstats
# from drills.fixed_optimization import optimize_with_fixed_script
# from pyfiglet import Figlet

def log(message):
    print('[DRiLLS {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + "] " + message)
    pass

def main():
    args = parse_arguments()
    with open('params.yml', 'r') as file:
        params = yaml.safe_load(file)
    params['library_file'] = args.library
    params['design_file'] = args.netlist
    params['cost_estimator'] = args.cost_function
    params['output_file'] = args.output
    if not os.path.exists(params['playground_dir']):
        os.mkdir(params['playground_dir'])
    params['playground_dir'] = os.path.join(params['playground_dir'], f"{os.path.basename(params['design_file'][:-2])}-{os.path.basename(params['cost_estimator'])}")
    if not os.path.exists(params['playground_dir']):
        os.mkdir(params['playground_dir'])
    # print(params)
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
    session_start = time.time()
    session_end = session_start
    final_min_cost = float('inf')
    while session_end - session_start <= 300:
        epoch_start = time.time()
        min_cost = float('inf')
        # phase 1 GA
        ga = GA(
            n=params['ga_n'],
            n_init=params['ga_n_init'],
            n_iter=params['ga_n_iter'],
            init_population=[[x['id'] for x in library.min_cell_map.values()]],
            dim_limit=[len(library.cell_map[gate_type]) for gate_type in library.gate_types],
            k_solution=5
        )
        ga_start = time.time()
        ga.run()
        ga_end = time.time()
        print(f"GA GENLIB takes {ga_end - ga_start:.2f} s")
        best_cell_maps, costs = ga.get_results()
        min_cell_map = None
        # write best genlibs
        best_cells_map = {gate_type: [] for gate_type in library.gate_types}
        for i, best_cell_map in enumerate(best_cell_maps):
            for gate_type, cell in best_cell_map.items():
                if cell not in best_cells_map[gate_type]:
                    best_cells_map[gate_type].append(cell)
            library.write_library_genlib(
                best_cell_map, 
                os.path.join(library.genlib_dir_path, f"library_{i}.genlib"),
            )
            if i == len(best_cell_maps) - 1:
                library.write_library_genlib(
                    best_cell_map, 
                    os.path.join(library.genlib_dir_path, "library.genlib"),
                )
                min_cost_ga_genlib = -costs[i]
                min_cell_map = best_cell_map
        library.write_library_genlib_all(
            dest=os.path.join(library.genlib_dir_path, "library_multi.genlib"),
            cell_map=best_cells_map
        )
        # sa
        sa_init_solution = []
        for i, (gate_type, cells) in enumerate(best_cells_map.items()):
            for j, cell in enumerate(cells):
                sa_init_solution.append(cell['cost'])
        sa = SA(
            init_solution=sa_init_solution, 
            init_cost=min_cost, 
            cell_map=best_cells_map,
            temp_h=4000000000, temp_l=0.1, 
            cooling_rate=0.5, num_iterations=500,
        )
        min_cell_map, min_cost_sa = sa.run()
        print(f"SA minimum cost: {min_cost_sa}, GA minimum cost: {min_cost_ga_genlib}")
        if min_cost_sa < min_cost_ga_genlib:
            library.write_library_genlib_all(
                cell_map=min_cell_map, 
                dest=os.path.join(library.genlib_dir_path, "library.genlib")
            )
        library.write_library_genlib_all(
            cell_map=min_cell_map, 
            dest=os.path.join(library.genlib_dir_path, "library_5.genlib")
        )
        # # phase 2 abc_ga
        # print(params['actions'])
        choice_commands = {'choice', 'choice2', 'dch', 'dch -f', 'dch -p', 'dch -fp', 'dc2 -p', 'dc2'}
        len_choices = sum(i in choice_commands for i in params['actions'])
        len_choice_commands = 4
        len_commands = len(params['actions'])
        dim_limit = [(0, len_commands-len_choices) for _ in range(params['abc_ga_seq_len'])]
        dim_limit = [(0, len_commands-len_choices) for _ in range(params['abc_ga_seq_len']-len_choice_commands)] + \
                    [(len_commands-len_choices-2, len_commands) for _ in range(len_choice_commands)]
        print(dim_limit)
        abc_ga = AbcGA(
            design_path=abc_session.preprocessed_design_path,
            output_file=params['output_file'],
            library_path=os.path.join(library.genlib_dir_path, "library.genlib"),
            actions=params['actions'],
            n_choice=len_choice_commands,
            choice_commands=choice_commands,
            mutation_rate=0.5,
            n=params['abc_ga_n'],
            n_init=params['abc_ga_n_init'],
            n_iter=params['abc_ga_n_iter'],
            dim_limit=dim_limit,
        )
        abc_ga.run()
        best_action_seqs, best_iteration, best_id, best_path, costs = abc_ga.get_results()
        abc_ga.output_netlist()
        # write best netlists
        min_cost = min(costs)
        print(f"abcGA min cost: {min_cost}")

        opt = PostOptimizor()
        # phase 3 post map
        # best_netlist, cost = opt.post_map(params['output_file'])
        # if cost < min_cost:
        #     with open(best_netlist, 'r') as src:
        #         with open(params['output_file'], 'w') as dest:
        #             dest.write(src.read())
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
            min_fanout_cost = float('inf')
            min_cost_fanout = -1
            for i, max_fanout in enumerate(range(2, 40, 2)):
                file_name = os.path.basename(params['output_file']).replace('.v', f"_{buf_cell_name}_fan_{max_fanout}.v")
                dest = os.path.join(buffer_temp_dir, file_name)
                opt.insert_buffers(params['output_file'], dest_path=dest, max_fanout=max_fanout, buf_cell_name=buf_cell_name)
                cost = cost_interface.get_cost(dest)
                # print(f"max fanout: {max_fanout}, cost: {cost}")
                if cost < min_cost:
                    min_cost = cost
                    min_idx = i
                    best_buffer_netlist = dest
                    print(f"buf update cost: {cost}")
                if cost < min_fanout_cost:
                    min_fanout_cost = cost
                    min_cost_fanout = max_fanout
            print(f"Buffer: {buf_cell_name} min cost: {min_fanout_cost:.6f} with max fanout: {min_cost_fanout}")
        if best_buffer_netlist:
            print("buf: ", best_buffer_netlist)
            with open(best_buffer_netlist, 'r') as src:
                with open(params['output_file'], 'w') as dest:
                    dest.write(src.read())
        end = time.time()
        # phase 5 gate sizing
        best_netlist, cost = opt.run_gate_sizing(params['output_file'])
        print(f"gate sizing: {best_netlist}, {cost}")
        if cost < min_cost:
            with open(best_netlist, 'r') as src:
                with open(params['output_file'], 'w') as dest:
                    dest.write(src.read())
        epoch_cost = cost_interface.get_cost(params['output_file'])
        final_min_cost = min(epoch_cost, final_min_cost)
        session_end = time.time()
        print(f"epoch cost: {epoch_cost}, epoch time elapsed: {session_end-epoch_start:.2f}, total time elpased: {session_end-session_start:.2f}")
    print(f"FINAL_min_cost: {final_min_cost}")

if __name__ == '__main__':
    # read args and yaml
    # cProfile.run("main()", "main_stats")

    # p = pstats.Stats("main_stats")
    # p.sort_stats("cumulative").print_stats()
    main()