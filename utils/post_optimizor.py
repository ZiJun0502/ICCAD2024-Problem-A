import re
from os.path import basename, join, exists
from os import mkdir
from queue import Queue
from collections import defaultdict
from DRiLLS.abcSession import abcSession
from utils.cost_interface import CostInterface
from utils.library import Library
from utils.config import Config
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

                temp_netlist_path = \
                    join(playground_dir, design_name.replace(".v", f"_{cell_name}.v"))
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
        buf_pattern = f"{net}_buf_{{}}"
        child_net = buf_pattern.format(0)
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
        return buffer_declarations, leaf_nets
    def count_fanout(self, netlist_path):
        with open(netlist_path, 'r') as file:
            lines = file.readlines()
        net_fanout = defaultdict(int)
        net_driving_line = defaultdict(list)
        
        # Regular expression to capture the nets in each line
        net_regex = re.compile(r'\.(?:A|B|Y)\(([^)]+)\)')
        for i, line in enumerate(lines):
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
        return net_fanout
    def insert_buffers(self, netlist_path, dest_path='', max_fanout=5, buf_cell_name=''):
        with open(netlist_path, 'r') as file:
            lines = file.readlines()
        net_fanout = defaultdict(int)
        net_driving_line = defaultdict(list)
        
        # Regular expression to capture the nets in each line
        net_regex = re.compile(r'\.(?:A|B|Y)\(([^)]+)\)')
        for i, line in enumerate(lines):
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
        # Inserting buffers and modifying lines
        for net, count in net_fanout.items():
            if count > max_fanout:
                buffer_declarations, leaf_nets = self.insert_buffers_(
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
        # # Write the modified netlist back
        # print(dest_path)
        # if dest_path:
        #     with open(dest_path, 'w') as file:
        #         file.writelines(modified_netlist + endmodule_str)
        # else:
        #     with open(netlist_path, 'w') as file:
        #         file.writelines(modified_netlist + endmodule_str)
