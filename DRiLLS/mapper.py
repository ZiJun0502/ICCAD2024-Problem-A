import json
import re
from numpy import percentile
from DRiLLS.abcSession import abcSession
from utils.cost_interface import CostInterface
class Mapper:
    def __init__(self):
        self.gate_types = []
        self.cell_map = {}
        self.min_cell_map = {}  # gate_type: cell_name
        self.min_cell_cost = {} # gate_type: min_cell_cost
        self.lib_information = {}
        self.cost_interface = CostInterface()
    def load_library(self, lib_file):
        with open(lib_file, 'r') as file:
            lib = json.load(file)
        for key, val in lib['information'].items():
            self.lib_information[key] = val
        for cell in lib['cells']:
            if cell['cell_type'] in self.cell_map:
                self.cell_map[cell['cell_type']].append(cell)
            else:
                self.gate_types.append(cell['cell_type'])
                self.cell_map[cell['cell_type']] = [cell]
        # self.get_lowest()
        # self.omit_large_cell()
        # self.write_library_genlib()
    def check_valid_remove(self, cell_to_remove):
        temp = self.min_cell_map.copy()
        del temp[cell_to_remove]
        complete_set = [['not', 'or'], ['not', 'and'], ['nand'], ['nor']]
        # check if map is still functional complete (if map's keys contain one of above set)
        for required_keys in complete_set:
            if all(key in temp for key in required_keys):
                return True
        return False

    def omit_large_cell(self):
        """ 
        Remove the cell with an excessively high cost.
        """
        avg_area = sum(self.min_cell_cost.values()) / len(self.min_cell_cost)
        print(avg_area)
        sorted_map = dict(sorted(self.min_cell_cost.items(), key=lambda item: -item[1]))

        cost_values = list(self.min_cell_cost.values())
        Q1 = percentile(cost_values, 25)
        Q3 = percentile(cost_values, 75)
        IQR = Q3 - Q1
        threshold = Q3 + 3 * IQR
        print(threshold)

        for cell in sorted_map:
            print(cell, self.min_cell_cost[cell])
            if self.min_cell_cost[cell] > threshold and self.check_valid_remove(cell):
                print(f"Removing cell: {cell}")
                del self.min_cell_cost[cell]
                del self.min_cell_map[cell]
                self.gate_types.remove(cell)
                

    def write_library_genlib(self):
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
        with open('library.genlib', 'w') as f:
            for gate_type in self.gate_types:
                for cell in self.cell_map.get(gate_type):
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
            f.write('GATE not_dummy    1  Y=!A;                   PIN * INV 1 999 1 0 1 0')
    def replace_dummy(self, src):
        lines = None
        with open(src, 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            if line.strip():  # Ensure the line is not empty or just whitespace
                if line.strip().startswith('not_dummy'):
                    # Extract the original A input net
                    original_a_net = line.split('.A(')[1].split(')')[0].strip()
                    # Construct the replacement line
                    gate_name = line[12:line.find('(')]
                    new_line = f"\t{self.min_cell_map['nand']}    {gate_name}(.A({original_a_net}), .B({original_a_net}), .Y({line.split()[-1].split('(')[-1].split(')')[0]}));\n"
                    # Append the modified line to the list
                    lines[i] = new_line
        modified_content = ''.join(lines)
        # print(modified_content)
        with open(src, 'w') as file:
            file.write(modified_content)


    def get_lowest(self):
        src = ""
        get_low_file = "./DRiLLS/get_low.v"
        with open(get_low_file, "r") as f:
            src = f.readlines()
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
                    gate_line = self._map_line(cell['cell_name'], inputs, output) + '\n'

                    src[j] = gate_line
                with open(get_low_file, "w") as f:
                    f.write("".join(src))

                cost = self.cost_interface.get_cost(get_low_file)# / (GATE_LINE_END - GATE_LINE_BEGIN)
                self.cell_map[gate_type][i]['cost'] = cost
                if cost < min_cost:
                    min_cost = cost
                    min_cell_name = cell['cell_name']
            self.min_cell_cost[gate_type] = min_cost
            self.min_cell_map[gate_type] = min_cell_name
    def _map_line(self, cell_name, inputs, output):
        return f"{cell_name} ( {', '.join(inputs)}, {output} );"
    def map(self, design_file, destination_file):
        with open(design_file, 'r') as file:
            netlist_lines = file.readlines()
        mapped_netlist = []
        for line in netlist_lines:
            line = line.strip()
            if any(line.startswith(gate_type) for gate_type in self.gate_types):
                ports = line[line.find('(')+1:line.find(')')].replace(' ','').split(',')
                inputs, output = ports[1:], ports[0]
                parts = line[:line.find('(')].split()
                gate = parts[0]
                # naive mapping
                choosen_cell = self.min_cell_map[gate]

                line = self._map_line(choosen_cell, inputs, output)
            mapped_netlist.append(line)

        # Join the mapped netlist into a single string
        mapped_netlist_str = "\n".join(mapped_netlist)
        with open(destination_file, 'w') as file:
            file.write(mapped_netlist_str)
# if __name__ == '__main__':
#     # Example usage:
#     design_file = "../release/netlists/design1.v"
#     lib_file = "../release/lib/lib1.json"

#     # Call the function with the design file and library file
#     mapper = Mapper()
#     mapper.load_library(lib_file)
#     mapper.map(design_file)
