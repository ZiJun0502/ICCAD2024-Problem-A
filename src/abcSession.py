import os
import re
import datetime
import numpy as np
import time
from subprocess import check_output
from cost_interface import CostInterface
from config import Config
from library import Library
class abcSession:
    """
    A class to represent a logic synthesis optimization session using ABC
    """
    def __init__(self):
        self.config = Config()
        self.library = Library()
        self.params = self.config.params

        self.cost_interface = CostInterface()

        self.preprocessed_design_path = os.path.join(self.params['playground_dir'], "design_preprocessed.v")
        self._pre_process_design()
        self.action_space_length = len(self.params['actions'])
        self.best_known_cost = (float('inf'), -1, -1)
        # logging
        self.log = None

    def _pre_process_design(self):
        """
        Preprocess the input design file to match abc input file format
        """
        with open(self.params['design_file'], 'r') as file:
            netlist_str = file.read()
        # remove gate names
        pattern = re.compile(r'(\w+\s+g\d+\s*)\((.*)\)')
        def replace_gate_name(match):
            gate_type_with_name = match.group(1)  # This includes both the gate type and gate name
            gate_type = gate_type_with_name.split()[0]  # Extract only the gate type (e.g., 'or', 'xnor')
            connections = match.group(2)  # This includes the connections inside the parentheses
            return f'{gate_type} ({connections})'

        processed_netlist = pattern.sub(replace_gate_name, netlist_str)
        # use preprocessed file afterwards
        self.current_netlist = self.preprocessed_design_path

        with open(self.preprocessed_design_path, 'w') as file:
            file.write(processed_netlist)

    def run_ga_genlib_all(self, design_path, library_paths, dests):
        # save first mapped network
        abc_command = f'read {design_path};'
        abc_command += 'strash;'
        # abc_command += 'fraig_store;'
        abc_command += f'read_library {library_paths[0]};'
        abc_command += 'map -a;'
        abc_command += f'write {dests[0]};'
        for i in range(1, len(library_paths)):
            # abc_command += 'fraig_restore;'
            # abc_command += 'fraig_store;'
            abc_command += f'read {design_path};'
            abc_command += 'strash;'
            abc_command += f'read_library {library_paths[i]};'
            abc_command += 'map -a;'
            abc_command += f'write {dests[i]};'

        proc = check_output([self.params['abc_binary'], '-c', abc_command])
        # print(proc.decode())
        for i in range(len(library_paths)):
            self.library.replace_dummy(dests[i])

        costs = [self.cost_interface.get_cost(netlist) for netlist in dests]
        return costs

    def run_ga_genlib(self, design_path, library_path, dest="ga_netlist.v"):
        # print(library_path)
        # if dest.startswith('./'):
            # dest = dest[2:]
        if dest == "ga_netlist.v":
            abc_mapped_output_netlist = os.path.join(self.params['playground_dir'], dest)
        else:
            abc_mapped_output_netlist = dest

        abc_command = f'read {design_path};'
        abc_command += 'strash;'
        abc_command += f'read_library {library_path};'
        abc_command += 'map -a;'
        abc_command += f'write {abc_mapped_output_netlist};'
        proc = check_output([self.params['abc_binary'], '-c', abc_command])
        self.library.replace_dummy(abc_mapped_output_netlist)
        # print(proc.decode())
        cost = self.cost_interface.get_cost(abc_mapped_output_netlist)
        return cost
    def unmap(self, src, dest, library_path):
        abc_command = f'read_library {os.path.join(self.library.genlib_dir_path, "init_library.genlib")};\n'
        abc_command += f'read -m {src};\n'
        abc_command += 'unmap;\n'
        abc_command += f'write {dest};\n'
        proc = check_output([self.params['abc_binary'], '-c', abc_command])
        # print(proc.decode())
    def forward_command(self, command, src, dest, library_path="", mapping=False):
        abc_command = ""
        abc_command += f'read {src};\n'
        abc_command += f'{command};\n'
        if mapping:
            abc_command  += f'read_library {library_path};\n'
            abc_command  += f'map -a;'
        abc_command += f'write {dest};\n'

        proc = check_output([self.params['abc_binary'], '-c', abc_command])
        try:
            self.library.replace_dummy(dest)
        except:
            pass
        # print(proc.decode())
        return proc

    def run_ga_abc_all(self, design_path, library_path, command_lists, dests=""):
        dests_unmapped = [dest.replace('.v', '_unmapped.v') for dest in dests]
        # save network aig
        abc_command = f'read {design_path};\n'
        abc_command += 'strash;\n'
        # abc_command += 'fraig_store;\n'
        abc_command += f'read_library {library_path};\n'
        # first chromosome
        abc_command += ';'.join(command_lists[0]) + ';\n'
        # abc_command += f'write {dests_unmapped[0]};'
        abc_command += 'map -a;\n'
        # abc_command += 'save;\n'
        abc_command += f'write {dests[0]};\n'
        for i in range(1, len(command_lists)):
            abc_command += f'read {design_path};\n'
            # abc_command += 'load;\n'
            abc_command += 'strash;\n'
            # abc_command += 'fraig_store;\n'
            abc_command += ';'.join(command_lists[i]) + ';\n'
            abc_command += 'map -a;\n'
            abc_command += f'write {dests[i]};\n'
        # proc = check_output([self.params['abc_binary'], '-c', abc_command])
        # try:
        start = time.time()
        proc = check_output([self.params['abc_binary'], '-c', abc_command])
        # print(proc.decode())
        end = time.time()
        # print(f"abc execute commands takes: {end-start:.2f}s")

        start = time.time()
        [self.library.replace_dummy(dest) for dest in dests]
        end = time.time()
        # print(f"replace dummy takes: {end-start:.2f}s")
        start = time.time()
        costs = [self.cost_interface.get_cost(dest) for dest in dests]
        end = time.time()
        # print(f"getting costs takes: {end-start:.2f}s")
        return costs
    # def run_ga_abc_all(self, design_path, library_path, command_lists, dests=""):
    #     dests_unmapped = [dest.replace('.v', '_unmapped.v') for dest in dests]
    #     # save network aig
    #     abc_command = f'read {design_path};\n'
    #     abc_command += 'strash;\n'
    #     # abc_command += 'fraig_store;\n'
    #     abc_command += f'read_library {library_path};\n'
    #     # first chromosome
    #     abc_command += ';'.join(command_lists[0]) + ';\n'
    #     # abc_command += f'write {dests_unmapped[0]};'
    #     abc_command += 'map -a;\n'
    #     abc_command += 'save;\n'
    #     abc_command += f'write {dests[0]};\n'
    #     for i in range(1, len(command_lists)):
    #         abc_command += 'load;\n'
    #         abc_command += 'strash;\n'
    #         # abc_command += 'fraig_store;\n'
    #         abc_command += ';'.join(command_lists[i]) + ';\n'
    #         abc_command += 'map -a;\n'
    #         abc_command += f'write {dests[i]};\n'
    #     # proc = check_output([self.params['abc_binary'], '-c', abc_command])
    #     # try:
    #     start = time.time()
    #     proc = check_output([self.params['abc_binary'], '-c', abc_command])
    #     # print(proc.decode())
    #     end = time.time()
    #     # print(f"abc execute commands takes: {end-start:.2f}s")

    #     start = time.time()
    #     [self.library.replace_dummy(dest) for dest in dests]
    #     end = time.time()
    #     # print(f"replace dummy takes: {end-start:.2f}s")
    #     start = time.time()
    #     costs = [self.cost_interface.get_cost(dest) for dest in dests]
    #     end = time.time()
    #     # print(f"getting costs takes: {end-start:.2f}s")
    #     return costs
    def run_ga_abc(self, design_path, library_path, commands, dest=""):
        dest_unmapped = dest.replace('.v', '_unmapped.v')

        abc_command = f'read {design_path};'
        abc_command += 'strash;' + ';'.join(commands) + ';'
        abc_command += f'write {dest_unmapped};'
        abc_command += f'read_library {library_path};'
        abc_command += 'map -a;'
        abc_command += f'write {dest};'
        # proc = check_output([self.params['abc_binary'], '-c', abc_command])
        # try:
        proc = check_output([self.params['abc_binary'], '-c', abc_command])
        # print(proc.decode())
        # except Exception as e:
        #     print(proc.decode())
        #     return None, None

        # print(proc.decode())
        self.library.replace_dummy(dest)

        cost = self.cost_interface.get_cost(dest)
        return cost
