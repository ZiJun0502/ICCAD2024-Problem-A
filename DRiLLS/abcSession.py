import os
import re
import datetime
import numpy as np
import time
from subprocess import check_output
from .features import extract_features
from utils.cost_interface import CostInterface
from utils.config import Config
from utils.library import Library
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

        cost = self.cost_interface.get_cost(abc_mapped_output_netlist)
        return cost
    def run_ga_abc(self, design_path, library_path, commands, dest=""):
        dest_unmapped = dest.replace('.v', '_unmapped.v')

        abc_command = f'read {design_path};'
        abc_command += 'strash;' + ';strash;'.join(commands) + ';'
        abc_command += f'write {dest_unmapped};'
        abc_command += f'read_library {library_path};'
        abc_command += 'map -a;'
        abc_command += f'write {dest};'
        # proc = check_output([self.params['abc_binary'], '-c', abc_command])
        proc = check_output([self.params['abc_binary'], '-c', abc_command])
        # print(proc)
        self.library.replace_dummy(dest)

        cost = self.cost_interface.get_cost(dest)
        return cost
