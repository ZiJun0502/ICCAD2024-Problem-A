from json import load
from utils.cost_interface import CostInterface
from numpy import percentile
from utils.config import Config
from os.path import join, exists
from os import mkdir
import random
class Library:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Library, cls).__new__(cls)
        return cls._instance
    def __init__(self, library_path=""):
        if not self._initialized:
            self.cell_map = {}
            self.min_cell_cost = {}
            self.min_cell_map = {}
            self.removed_cells = []
            # self.lib_information = {}
            self.gate_types = []
            self.cost_interface = CostInterface()

            self.config = Config()
            self.load_library(library_path)
            self.get_lowest()
            self.omit_large_cell()
            self.genlib_dir_path = join(self.config.params['playground_dir'], "lib")
            if not exists(self.genlib_dir_path):
                mkdir(self.genlib_dir_path)
            self.write_library_genlib_all(join(self.genlib_dir_path, 'init_library.genlib'))
            self.write_library_genlib(
                chosen_cell_map = self.min_cell_map,
                dest = join(self.genlib_dir_path, 'library_min.genlib'),
            )
            self._initialized = True
    def load_library(self, library_path):
        with open(library_path, 'r') as file:
            lib = load(file)
        # for key, val in lib['information'].items():
        #     self.lib_information[key] = val
        for i, cell in enumerate(lib['cells']):
            cell = {k: cell[k] for k in ["cell_type", "cell_name"] if k in cell}

            if cell['cell_type'] in self.cell_map:
                cell['id'] = len(self.cell_map[cell['cell_type']])
                self.cell_map[cell['cell_type']].append(cell)
            else:
                cell['id'] = 0
                self.gate_types.append(cell['cell_type'])
                self.cell_map[cell['cell_type']] = [cell]
    def get_lowest(self):
        src = ""
        get_low_file = "./DRiLLS/get_low.v"
        playground_get_low_file = join(self.config.params['playground_dir'], "get_low.v")
        with open(get_low_file, "r") as f:
            src = f.readlines()
            with open(playground_get_low_file, "w") as ff:
                ff.write("\n".join(src))
        GATE_LINE_BEGIN, GATE_LINE_END = 4, 14
        for gate_type in self.gate_types:
            min_cost = float('inf')
            min_cell_name = ""
            for i, cell in enumerate(self.cell_map.get(gate_type)):
                # write fake verilog file to check the cost of each gate
                for j in range(GATE_LINE_BEGIN, GATE_LINE_END):
                    inputs = [f'a{j}', f'b{j}']
                    output = f'o{j}'
                    if gate_type in ['buf', 'not']:
                        inputs.pop()
                    gate_line = self.format_line(cell['cell_name'], inputs, output) + '\n'

                    src[j] = gate_line
                with open(playground_get_low_file, "w") as f:
                    f.write("".join(src))

                cost = self.cost_interface.get_cost(playground_get_low_file)# / (GATE_LINE_END - GATE_LINE_BEGIN)
                self.cell_map[gate_type][i]['cost'] = cost
                if cost < min_cost:
                    min_cost = cost
                    min_cell = cell
            self.min_cell_cost[gate_type] = min_cost
            self.min_cell_map[gate_type] = min_cell
        # for g, c in self.min_cell_cost.items():
        #     print(self.min_cell_map[g]['cell_name'], c)
    def omit_large_cell(self):
        """ 
        Remove the cell with an excessively high cost.
        """
        avg_area = sum(self.min_cell_cost.values()) / len(self.min_cell_cost)
        # print(avg_area)
        sorted_map = dict(sorted(self.min_cell_cost.items(), key=lambda item: -item[1]))

        cost_values = list(self.min_cell_cost.values())
        Q1 = percentile(cost_values, 25)
        Q3 = percentile(cost_values, 75)
        IQR = Q3 - Q1
        threshold = Q3 + 3 * IQR
        # print(threshold)

        for cell in sorted_map:
            # print(cell, self.min_cell_cost[cell])
            if self.min_cell_cost[cell] > threshold and self.check_valid_remove(cell):
                print(f"Removing cell: {cell}")
                self.removed_cells.append(cell)
                # del self.min_cell_cost[cell]
                # del self.min_cell_map[cell]
                # self.gate_types.remove(cell)
    def check_valid_remove(self, cell_to_remove):
        temp = self.min_cell_map.copy()
        del temp[cell_to_remove]
        complete_set = [['not', 'or'], ['not', 'and'], ['nand'], ['nor']]
        # check if map is still functional complete (if map's keys contain one of above set)
        for required_keys in complete_set:
            if all(key in temp for key in required_keys):
                return True
        return False
    def format_line(self, cell_name, inputs, output):
        return f"{cell_name} ( {', '.join(inputs)}, {output} );"
    def replace_dummy(self, src):
        lines = None
        with open(src, 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if line.strip():  # Ensure the line is not empty or just whitespace
                if line.strip().startswith('not_dummy'):
                    # print("replacing dummy")
                    # Extract the original A input net
                    original_a_net = line.split('.A(')[1].split(')')[0].strip()
                    # Construct the replacement line
                    gate_name = line[12:line.find('(')]
                    if 'nand' in self.removed_cells:
                        new_line = f"\t{self.min_cell_map['nor']['cell_name']}    {gate_name}(.A({original_a_net}), .B({original_a_net}), .Y({line.split()[-1].split('(')[-1].split(')')[0]}));\n"
                    else:
                        new_line = f"\t{self.min_cell_map['nand']['cell_name']}    {gate_name}(.A({original_a_net}), .B({original_a_net}), .Y({line.split()[-1].split('(')[-1].split(')')[0]}));\n"
                    # Append the modified line to the list
                    lines[i] = new_line
        modified_content = ''.join(lines)
        # print(modified_content)
        with open(src, 'w') as file:
            file.write(modified_content)
    # def write_sa_genlib(self, solution):
        
    def write_library_genlib(self, chosen_cell_map, dest):
        cell_function = {
            "and": "Y=A*B;",
            "not": "Y=!A;",
            "buf": "Y=A;",
            "or": "Y=A+B;",
            "nand": "Y=!(A*B);",
            "nor": "Y=!(A+B);",
            "xor": "Y=(A*!B)+(!A*B);",
            "xnor": "Y=(A*B+!A*!B);",
            "zero": "Y=CONST0;",
            "one": "Y=CONST1;"
        }
        cell_num_pins = {
            "and": 2,
            "not": 1,
            "buf": 1,
            "or": 2,
            "nand": 2,
            "nor": 2,
            "xor": 2,
            "xnor": 2,
            "zero": 0,
            "one": 0
        }
        cell_phase = {
            "and": "NONINV",
            "not": "INV",
            "buf": "NONINV",
            "or": "NONINV",
            "nand": "INV",
            "nor": "INV",
            "xor": "UNKNOWN",
            "xnor": "UNKNOWN",
            "zero": "",
            "one": ""
        }
        input_load, max_load = 1, 999
        rb_delay, rf_delay = 1, 0 
        fb_delay, ff_delay = 1, 0 
        with open(dest, 'w') as f:
            for gate_type in self.gate_types:
                if gate_type in self.removed_cells:
                    if gate_type == 'not':
                        f.write('GATE not_dummy    0.8  Y=!A;                   PIN * INV 1 999 1 0 1 0\n')
                else:
                    # cell_area = random.random()
                    cell = chosen_cell_map[gate_type]
                    function = cell_function.get(gate_type)
                    phase = cell_phase.get(gate_type, "UNKNOWN")
                    # gate_name = f"GATE {cell['cell_name']:<8} {cell['cost']}  {function}"
                    gate_name = f"GATE {cell['cell_name']:<8} {cell['cost']}  {function}"
                    # pin_info = f"PIN Y {phase} {input_load} {max_load} {rb_delay} {rf_delay} {fb_delay} {ff_delay}\n" if phase else ""
                    pin_info = f"PIN * {phase} {input_load} {max_load}\n {rb_delay} {rf_delay}\n {fb_delay} {ff_delay}\n" if phase else ""
                    # if(cell_num_pins[gate_type] == 2):
                    #     pin_info += f"PIN B {phase} {input_load} {max_load}\n {rb_delay} {rf_delay}\n {fb_delay} {ff_delay}\n" if phase else ""
                    f.write(f"{gate_name:<40}\n{pin_info}\n")
            f.write('GATE ZERO      1  Y=CONST0;\n')
            f.write('GATE ONE       1  Y=CONST1;\n')
            # f.write('GATE not_dummy    0.8  Y=!A;                   PIN * INV 1 999 1 0 1 0\n')

    def write_library_genlib_all(self, dest, cell_map={}):
        if not cell_map:
            cell_map = self.cell_map
        cell_function = {
            "and": "Y=A*B;",
            "not": "Y=!A;",
            "buf": "Y=A;",
            "or": "Y=A+B;",
            "nand": "Y=!(A*B);",
            "nor": "Y=!(A+B);",
            "xor": "Y=(A*!B)+(!A*B);",
            "xnor": "Y=(A*B+!A*!B);",
            "zero": "Y=CONST0;",
            "one": "Y=CONST1;"
        }
        cell_num_pins = {
            "and": 2,
            "not": 1,
            "buf": 1,
            "or": 2,
            "nand": 2,
            "nor": 2,
            "xor": 2,
            "xnor": 2,
            "zero": 0,
            "one": 0
        }
        cell_phase = {
            "and": "NONINV",
            "not": "INV",
            "buf": "NONINV",
            "or": "NONINV",
            "nand": "INV",
            "nor": "INV",
            "xor": "UNKNOWN",
            "xnor": "UNKNOWN",
            "zero": "",
            "one": ""
        }
        input_load, max_load = 1, 999
        rb_delay, rf_delay = 1, 0 
        fb_delay, ff_delay = 1, 0 
        with open(dest, 'w') as f:
            for gate_type in self.gate_types:
                if gate_type in self.removed_cells:
                    if gate_type == 'not':
                        f.write('GATE not_dummy    0.8  Y=!A;                   PIN * INV 1 999 1 0 1 0\n')
                        continue
                for cell in cell_map.get(gate_type):
                    function = cell_function.get(gate_type)
                    phase = cell_phase.get(gate_type, "UNKNOWN")
                    gate_name = f"GATE {cell['cell_name']:<8} {cell['cost']}  {function}"
                    # pin_info = f"PIN Y {phase} {input_load} {max_load} {rb_delay} {rf_delay} {fb_delay} {ff_delay}\n" if phase else ""
                    pin_info = f"PIN A {phase} {input_load} {max_load}\n {rb_delay} {rf_delay}\n {fb_delay} {ff_delay}\n" if phase else ""
                    if(cell_num_pins[gate_type] == 2):
                        pin_info += f"PIN B {phase} {input_load} {max_load}\n {rb_delay} {rf_delay}\n {fb_delay} {ff_delay}\n" if phase else ""
                    f.write(f"{gate_name:<40}\n{pin_info}\n")

            f.write('GATE ZERO      1  Y=CONST0;\n')
            f.write('GATE ONE       1  Y=CONST1;\n')
            f.write('GATE not_dummy    0.8  Y=!A;                   PIN * INV 1 999 1 0 1 0')