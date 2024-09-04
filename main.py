import os
import shutil
import random
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import datetime
import numpy as np
import time
import copy
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
from ga.gate_sizing_ga import GateSizingGA

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
    params['output_path'] = args.output
    if not os.path.exists(params['playground_dir']):
        os.mkdir(params['playground_dir'])
    params['playground_dir'] = os.path.join(params['playground_dir'], f"{os.path.basename(params['design_file'][:-2])}-{os.path.basename(params['cost_estimator'])}")
    if os.path.exists(params['playground_dir']):
        shutil.rmtree(params['playground_dir'])
    if not os.path.exists(params['playground_dir']):
        os.mkdir(params['playground_dir'])
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
    session_min_cost = float('inf')
    epoch = 0
    first = True
    while session_end - session_start <= 600:
        epoch_start = time.time()
        epoch_cost = float('inf')
        print([(x['cell_name'], x['cost']) for x in library.min_cell_map.values()])
        """
        Phase 1: GA-genlib
        """
        print([x['cell_name'] for x in library.min_cell_map.values()])
        # if first:
        n_iter = params['ga_n_iter']
        # else:
            # n_iter = random.randint(10, params['ga_n_iter'])
            # n_iter = 10
        ga = GA(
            n=params['ga_n'],
            n_init=params['ga_n_init'],
            n_iter=n_iter,
            mutation_decay=False,
            output_path=params['output_path'],
            init_population=[[x['id'] for x in library.min_cell_map.values()]],
            session_min_cost=session_min_cost,
            dim_limit=[len(library.cell_map[gate_type]) for gate_type in library.gate_types],
            k_solution=5
        )
        start = time.time()
        ga.run()
        best_cell_maps, ga_costs = ga.get_results()
        end = time.time()
        print(f"GA-genlib takes {end - start:.2f} s")
        print(f"GA-genlib cost: {ga_costs[-1]}")
        ga_genlib_min_cost = ga_costs[-1]
        epoch_cost = ga_genlib_min_cost
        if epoch_cost < session_min_cost:
            session_min_cost = epoch_cost
            print(f"New session min cost: {session_min_cost}")
        ga_min_cell_map = None
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
                ga_min_cell_map = best_cell_map
        library.write_library_genlib_all(
            dest=os.path.join(library.genlib_dir_path, "library_multi.genlib"),
            cell_map=best_cells_map
        )
        # # print(min_cell_map)
        """
        Phase 2: SA
        """
        sa_init_solution = []
        for i, (gate_type, cells) in enumerate(best_cells_map.items()):
            for j, cell in enumerate(cells):
                sa_init_solution.append(cell['cost'])
        sa = SA(
            init_solution=sa_init_solution, 
            init_cost=epoch_cost, 
            cell_map=copy.deepcopy(best_cells_map),
            # cell_map=best_cells_map,
            temp_h=4000000000, temp_l=0.1, 
            cooling_rate=0.5, num_iterations=params['sa_n_iter'],
        )
        min_cell_map, sa_min_cost = sa.run()
        print(f"SA minimum cost: {sa_min_cost}, GA minimum cost: {ga_genlib_min_cost}")
        if sa_min_cost < ga_genlib_min_cost:
            library.write_library_genlib_all(
                cell_map=min_cell_map, 
                dest=os.path.join(library.genlib_dir_path, "library.genlib")
            )
            epoch_cost = sa_min_cost
        library.write_library_genlib_all(
            cell_map=min_cell_map, 
            dest=os.path.join(library.genlib_dir_path, "library_5.genlib")
        )
        """
        Phase 3: Abc_GA
        """
        # print(params['actions'])
        choice_commands = {'choice', 'choice2', 'dch', 'dch -f', 'dch -p', 'dch -fp', 'dc2 -p', 'dc2'}
        len_choices = sum(i in choice_commands for i in params['actions'])
        # for i, action in enumerate(params['actions']):
        #     if action in choice_commands:
        #         params['actions'][i] = f"{action};st;"
        len_choice_commands = 4
        len_commands = len(params['actions'])
        # dim_limit = [(0, len_commands-len_choices) for _ in range(params['abc_ga_seq_len'])]
        dim_limit = [(0, len_commands-len_choices) for _ in range(params['abc_ga_seq_len']-len_choice_commands)] + \
                    [(len_commands-len_choices-2, len_commands) for _ in range(len_choice_commands)]
        dim_limit = [(0, len_commands) for _ in range(params['abc_ga_seq_len'])]
        print(dim_limit)
        abc_ga = AbcGA(
            design_path=abc_session.preprocessed_design_path,
            output_path=params['output_path'],
            library_path=os.path.join(library.genlib_dir_path, "library.genlib"),
            actions=params['actions'],
            seq_len=params['abc_ga_seq_len'],
            n_choice=len_choice_commands,
            choice_commands=choice_commands,
            mutation_rate=0.5,
            mutation_decay=False,
            n=params['abc_ga_n'],
            n_init=params['abc_ga_n_init'],
            n_iter=params['abc_ga_n_iter'],
            dim_limit=dim_limit,
            session_min_cost=session_min_cost,
        )
        abc_ga.run()
        best_action_seqs, best_iteration, best_id, best_path, abc_ga_costs = abc_ga.get_results()
        # abc_ga_best_netlist = os.join(params['playground_dir'], "abc_ga_best_netlist.v")
        # abc_ga.output_netlist(abc_ga_best_netlist)
        # write best netlists
        abc_ga_min_cost = min(abc_ga_costs)
        print(f"Abc-GA cost: {abc_ga_min_cost}")
        epoch_cost = min(epoch_cost, abc_ga_min_cost)
        if epoch_cost < session_min_cost:
            session_min_cost = epoch_cost
            print(f"New session min cost: {session_min_cost}")
        abc_ga_best_cost = cost_interface.get_cost(abc_ga.best_path)
        print(f"abc_ga best path: {abc_ga.best_path}, cost: {abc_ga_best_cost}")
        assert  abc_ga_best_cost == abc_ga_min_cost
        # print(f"abcGA min cost: {epoch_cost}")

        # ASSUME that the min epoch cost is guarenteed to be abc_ga best cost.
        epoch_best_netlist = abc_ga.best_path

        opt = PostOptimizor()
        # phase 3 post map
        # best_netlist, cost = opt.post_map(params['output_path'])
        # if cost < epoch_cost:
        #     with open(best_netlist, 'r') as src:
        #         with open(params['output_path'], 'w') as dest:
        #             dest.write(src.read())
        """
        Phase 4: Buffer Insertion
        """
        best_buffer_netlist = ""
        # find best buffer to insert
        start = time.time()
        best_buffer_netlist, buf_min_cost = opt.run_insert_buffers(
            design_path=abc_ga.best_path,
            buf_cells=library.cell_map['buf']
        )
        end = time.time()
        print(f"Buffer Insertion cost: {buf_min_cost}")
        if buf_min_cost < epoch_cost:
            epoch_cost = buf_min_cost
            epoch_best_netlist = best_buffer_netlist
        if epoch_cost < session_min_cost:
            session_min_cost = epoch_cost
            print(f"New session min cost: {session_min_cost}")
            with open(best_buffer_netlist, 'r') as src:
                with open(params['output_path'], 'w') as dest:
                    dest.write(src.read())
        assert cost_interface.get_cost(best_buffer_netlist) == buf_min_cost
        """
        Phase 5: Gate Sizing
        """
        start = time.time()
        best_gate_sizing_netlist, gs_min_cost = opt.run_gate_sizing(best_buffer_netlist)
        end = time.time()
        print(f"Gate Sizing cost: {gs_min_cost}")
        epoch_cost = min(epoch_cost, gs_min_cost)
        if epoch_cost < session_min_cost:
            session_min_cost = epoch_cost
            print(f"New session min cost: {session_min_cost}")
            with open(best_gate_sizing_netlist, 'r') as src:
                with open(params['output_path'], 'w') as dest:
                    dest.write(src.read())
        assert cost_interface.get_cost(best_gate_sizing_netlist) == gs_min_cost
        """
        Check cost
        """
        epoch_min_cost = cost_interface.get_cost(params['output_path'])
        print(f"session minimum: {session_min_cost}, current minimum: {epoch_min_cost}")
        assert session_min_cost == epoch_min_cost
        # session_min_cost = min(epoch_min_cost, session_min_cost)
        session_end = time.time()
        print(f"epoch {epoch} cost: {epoch_cost}, epoch time elapsed: {session_end-epoch_start:.2f}, total time elpased: {session_end-session_start:.2f}")
        print()
        first = False
        epoch += 1
    print(f"FINAL_min_cost: {session_min_cost}")

if __name__ == '__main__':
    # read args and yaml
    # cProfile.run("main()", "main_stats")

    # p = pstats.Stats("main_stats")
    # p.sort_stats("cumulative").print_stats()
    main()