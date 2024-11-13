import re
from os.path import basename, join, exists
from os import mkdir
from queue import Queue
from collections import defaultdict
from abcSession import abcSession
from cost_interface import CostInterface
from library import Library
from config import Config
buf_gate_count = 0
class PostOptimizor:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PostOptimizor, cls).__new__(cls)
        return cls._instance
    def __init__(self, k_genlib=5):
        if not self._initialized:
            self.library = Library()
            self.config = Config()
            self.cost_interface = CostInterface()
            self.abcSession = abcSession()
            self.k_genlib = k_genlib

            self.buffer_temp_dir = join(self.config.params['playground_dir'], "buffer")
            if not exists(self.buffer_temp_dir):
                mkdir(self.buffer_temp_dir)

            self.chain_temp_dir = join(self.config.params['playground_dir'], "chain")
            if not exists(self.chain_temp_dir):
                mkdir(self.chain_temp_dir)

            self._initialized = True

    def run_gate_sizing(self, design_path, genlib_path="", cell_map={}):
        playground_dir = join(self.config.params['playground_dir'], "gateSizing")
        if not exists(playground_dir):
            mkdir(playground_dir)
        design_name = basename(design_path)
        if not cell_map:
            cell_map = self.library.cell_map
        with open(design_path, 'r') as file:
            original_netlist = file.readlines()
        min_cost_path = design_path
        min_cost = float('inf')
        for gate_type, cell_list in cell_map.items():
            # print(gate_type, [cell_info['cell_name'] for cell_info in cell_list])
            cur_gate_min_cost = float('inf')
            for cell_info in cell_list:
                cell_name = cell_info['cell_name']
                modified_netlist = []
                
                for line in original_netlist:
                    if line.strip().startswith(gate_type):
                        line_cell = line.strip().split()[0]
                        modified_line = line.replace(line_cell, cell_name, 1)
                        modified_netlist.append(modified_line)
                    else:
                        modified_netlist.append(line)

                temp_netlist_path = join(playground_dir, design_name.replace(".v", f"_{cell_name}.v"))
                with open(temp_netlist_path, 'w') as temp_file:
                    temp_file.writelines(modified_netlist)
                cost = self.cost_interface.get_cost(temp_netlist_path)
                if cost < min_cost:
                    min_cost_path = temp_netlist_path
                    min_cost = cost
        return min_cost_path, min_cost
                # print(f"{cell_name}: {cost}", end=', ')
            # print()
        # print(f"original cost: {self.cost_interface.get_cost(design_path)}")
        # print(f"after gate-sizing cost: {min_cost}")
    # def post_area_search(self):
    #     """
    #     For each .genlib solutions, adjust area value for each cell [1-5]
    #     """
    #     for i in range(5):
    #         genlib_path = f"./lib/library_{i}.genlib"
    def post_map(self, design_path):
        min_cost = float('inf')
        min_iter = -1
        min_design_path = ""
        unmapped_design_path = design_path.replace('.v', '_unmapped.v')
        library_path = join(self.library.genlib_dir_path, "library.genlib")
        self.abcSession.unmap(design_path, unmapped_design_path, library_path)
        for i in range(6):
            genlib_path = join(self.library.genlib_dir_path, f"library_{i}.genlib")
            design_dest = design_path.replace(".v", f"_{i}.v")
            cost = self.abcSession.run_ga_genlib(unmapped_design_path, genlib_path, design_dest)
            print(f"design: {design_dest}, cost {cost}")
            if cost < min_cost:
                min_cost = cost
                min_iter = i
                min_design_path = design_dest
        return f"{min_design_path}", min_cost


    def insert_buffers_(self, net, fanout, max_fanout, buf_cell_name):
        global buf_gate_count
        # print(net, buf_gate_count)
        buf_pattern = f"{net}_buf_{{}}"
        child_net = buf_pattern.format(0)
        new_buf_wire_names = [child_net]
        # queue storing leaf nets
        leaf_nets = Queue()
        parent_net = child_net
        # buf_cell_name = self.library.min_cell_map['buf']['cell_name']
        buffer_declarations = [f"{buf_cell_name} b{buf_gate_count:05}(.A({net}), .Y({child_net}));\n"]
        buf_gate_count += 1
        buf_net_child_id = 1
        buf_net_parent_id = 0
        end = False
        while True:
            if not leaf_nets.empty():
                leaf_nets.get()
            for j in range(max_fanout):
                child_net = buf_pattern.format(buf_net_child_id)
                new_buf_wire_names.append(child_net)
                leaf_nets.put(child_net)
                buffer_declaration = f"{buf_cell_name} b{buf_gate_count:05}(.A({parent_net}), .Y({child_net}));\n"
                buffer_declarations.append(buffer_declaration)
                buf_gate_count += 1
                buf_net_child_id += 1
                if leaf_nets.qsize() == fanout:
                    end = True
                    break
            if end:
                break
            buf_net_parent_id += 1
            parent_net = buf_pattern.format(buf_net_parent_id)
        leaf_nets = [leaf_nets.get() for i in range(leaf_nets.qsize())]
        return buffer_declarations, leaf_nets, new_buf_wire_names
    def insert_buffers(self, netlist_path, dest_path='', max_fanout=5, buf_cell_name=''):
        with open(netlist_path, 'r') as file:
            lines = file.readlines()
        net_fanout = defaultdict(int)
        net_driving_line = defaultdict(list)
        wire_declaration_line = -1
        # Regular expression to capture the nets in each line
        net_regex = re.compile(r'\.(?:A|B|Y)\(([^)]+)\)')
        for i, line in enumerate(lines):
            if len(line.strip()) and line.strip()[-1] == ';' and wire_declaration_line == -1:
                wire_declaration_line = i
            # Find all nets in the current line
            matches = net_regex.findall(line)
            if len(matches) == 2:
                A, Y = matches
                net_fanout[A] += 1
                net_driving_line[A].append(i)
            elif len(matches) == 3:
                A, B, Y = matches
                net_fanout[A] += 1
                net_fanout[B] += 1
                net_driving_line[A].append(i)
                net_driving_line[B].append(i)
        modified_netlist, endmodule_str = lines[:-3], lines[-3:]

        # buf_cell_name = self.library.min_cell_map['buf']['cell_name']
        global buf_gate_count
        buf_gate_count = 0
        buf_wire_names = []
        # Inserting buffers and modifying lines
        for net, count in net_fanout.items():
            if count > max_fanout:
                buffer_declarations, leaf_nets, cur_buf_wire_names = self.insert_buffers_(
                    net=net,
                    fanout=count,
                    max_fanout=max_fanout,
                    buf_cell_name=buf_cell_name
                )
                
                # Distribute fanout across buffer stages
                for i, line_index in enumerate(net_driving_line[net]):
                    modified_netlist[line_index] = re.sub(rf'\.A\({net}\)', f'.A({leaf_nets[i]})', modified_netlist[line_index])
                    modified_netlist[line_index] = re.sub(rf'\.B\({net}\)', f'.B({leaf_nets[i]})', modified_netlist[line_index])
                modified_netlist += buffer_declarations
                buf_wire_names += cur_buf_wire_names
        wire_declaration_block = self.get_wire_declaration_str(buf_wire_names)
        modified_netlist.insert(wire_declaration_line + 1, wire_declaration_block)
        # # Write the modified netlist back
        # print(dest_path)
        if dest_path:
            with open(dest_path, 'w') as file:
                file.writelines(modified_netlist + endmodule_str)
        # else:
        #     with open(netlist_path, 'w') as file:
        #         file.writelines(modified_netlist + endmodule_str)
    def run_insert_buffers(self, design_path, buf_cells):
        min_cost = float('inf')
        min_cost_cell = ""
        min_cost_fanout = -1
        for buf_cell in buf_cells:
            buf_cell_name = buf_cell['cell_name']
            min_fanout_cost = float('inf')
            min_cost_fanout = -1
            for i, max_fanout in enumerate(range(2, 40, 2)):
                file_name = basename(design_path).replace('.v', f"_{buf_cell_name}_fan_{max_fanout}.v")
                dest = join(self.buffer_temp_dir, file_name)
                self.insert_buffers(design_path, dest_path=dest, max_fanout=max_fanout, buf_cell_name=buf_cell_name)
                cost = self.cost_interface.get_cost(dest)
                # print(f"max fanout: {max_fanout}, cost: {cost}")
                if cost < min_cost:
                    min_cost = cost
                    min_cost_cell = buf_cell_name
                    min_cost_fanout = max_fanout
                    best_buffer_netlist = dest
                if cost < min_fanout_cost:
                    min_fanout_cost = cost
                    min_cost_fanout = max_fanout
            print(f"Buffer: {buf_cell_name} min cost: {min_fanout_cost:.6f} with max fanout: {min_cost_fanout}")
        # print(f"Buffer min cost: {min_cost}, with buf cell: {min_cost_cell}, with fanout: {min_cost_fanout}")
        return best_buffer_netlist, min_cost
    def get_wire_declaration_str(self, wires):
        """
        Given a list of wire names, 
        return proper wire declaration lines for verilog file
        """
        wire_declaration_block = ""
        if len(wires):
            def chunk_list(lst, chunk_size):
                for i in range(0, len(lst), chunk_size):
                    yield lst[i:i + chunk_size]

            # Generate the wire declaration block with a maximum of 10 nets per line
            chunk_size = 30  # Maximum nets per line
            wire_declaration_lines = [", ".join(chunk) for chunk in chunk_list(wires, chunk_size)]
            wire_declaration_block = "  wire " + ",\n    ".join(wire_declaration_lines) + ";\n"
        return wire_declaration_block
    def _get_nets(self, design_str):
        selected_wire = ""
        pi_nets = []
        for i in range(len(design_str)):
            line = design_str[i].strip()
            # print(line)
            if line.startswith("input"):
                line = line[5:].strip().replace(" ", "")
                end = False
                while True:
                    if line[-1] == ';':
                        end = True
                    line = line[:-1]
                    line_nets = line.split(",")
                    if not selected_wire:
                        selected_wire = line_nets[0]
                    pi_nets += line_nets
                    if end:
                        break
                    i += 1
                    line = design_str[i].strip().replace(" ", "")
                    # print(line, "Last: ", line[-5:])
                return selected_wire, pi_nets
        return None, None
    def run_chain_insertion(self, design_path, chain_len=10):
        """
        insert chain of and gates starts with constant 1,
        reduce the heuristic dynamic power value
        """
        with open(design_path, 'r') as f:
            design_str = f.readlines()

        net, pi_nets = self._get_nets(design_str)
        # print(net, pi_nets)
        new_chain_nets = [f"{net}_chain_{i}" for i in range(chain_len)]
        xnor_cell = self.library.min_cell_map['xnor']['cell_name']
        and_cell = self.library.min_cell_map['and']['cell_name']
        # constant 1
        chains = [f"{xnor_cell} c00000(.A({net}), .B({net}), .Y({new_chain_nets[0]}));\n"]
        # and chain
        chains += [f"{and_cell} c{i+1:05}(.A({new_chain_nets[i]}), .B({new_chain_nets[i]}), .Y({new_chain_nets[i+1]}));\n" for i in range(chain_len-1)]
        # now we need to AND pi with out chain constant 1,
        num_pi = len(pi_nets)
        # print(pi_nets)
        chains += [f"{and_cell} ca{i:05}(.A({pi_net}), .B({new_chain_nets[-1]}), .Y({pi_net}_chain));\n" for i, pi_net in enumerate(pi_nets)]


        # all lines with pi in it should be renamed n2 -> n2_chain
        net_driving_line = {pi_net: [] for pi_net in pi_nets}
        # where new pi wires should be declared
        wire_declaration_line = -1
        new_pi_nets = [f"{pi_net}_chain" for pi_net in pi_nets]
        
        net_regex = re.compile(r'\.(?:A|B|Y)\(([^)]+)\)')
        for i, line in enumerate(design_str):
            if len(line.strip()) and line.strip()[-1] == ';' and wire_declaration_line == -1:
                wire_declaration_line = i
            # Find all nets in the current line
            matches = net_regex.findall(line)
            if len(matches) == 2:
                A, Y = matches
                if A in net_driving_line:
                    net_driving_line[A].append(i)
            elif len(matches) == 3:
                A, B, Y = matches
                if A in net_driving_line:
                    net_driving_line[A].append(i)
                if B in net_driving_line:
                    net_driving_line[B].append(i)
        # print(net_driving_line)

        # modified_netlist = design_str[:]
        modified_netlist, endmodule_str = design_str[:-3], design_str[-3:]
        # replace pi in each lines with pi_chain
        for pi_net in net_driving_line:
            for i, line_index in enumerate(net_driving_line[pi_net]):
                modified_netlist[line_index] = re.sub(rf'\.A\({pi_net}\)', f'.A({pi_net}_chain)', modified_netlist[line_index])
                modified_netlist[line_index] = re.sub(rf'\.B\({pi_net}\)', f'.B({pi_net}_chain)', modified_netlist[line_index])

        # declare new wires
        combined_new_nets = new_pi_nets + new_chain_nets
        
        wire_declaration_block = self.get_wire_declaration_str(combined_new_nets)
        modified_netlist.insert(wire_declaration_line + 1, wire_declaration_block)

        # add chain to netlist
        modified_netlist += chains
        # write to file and get cost
        file_name = basename(design_path).replace('.v', f"_chain.v")
        dest = join(self.chain_temp_dir, file_name)
        with open(dest, 'w') as file:
            file.writelines(modified_netlist + endmodule_str)
        cost = self.cost_interface.get_cost(dest)
        print(dest, cost)

        return dest, cost
        