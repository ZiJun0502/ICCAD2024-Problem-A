from utils.cost_interface import CostInterface
from utils.library import Library
from utils.config import Config
from utils.post_optimizor import PostOptimizor
import yaml
class Params:
    def __init__(self):
        self.library = "./release/lib/lib1.json"
        self.cost_estimator_path = "./release/cost_estimators/cost_estimator_1"
        self.netlist = "./release/netlists/design4.v"
        self.output = "./a.v"
        with open('params.yml', 'r') as file:
            self.params = yaml.safe_load(file)
        for key, value in self.params.items():
            setattr(self, key, value)
        self.params['library_file'] = self.library
        self.params['design_file'] = self.netlist
        self.params['cost_estimator'] = self.cost_estimator_path
        self.params['output_file'] = self.output
params = Params()

config = Config(
    params=params.params
)
cost_interface = CostInterface(
    cost_estimator_path=params.cost_estimator_path,
    library_path=params.library                           
)
library = Library(
    library_path=params.library                           
)

# a = CostInterface("./release/cost_estimators/cost_estimator_6", "./release/lib/lib1.json")
# a = PostOptimizor()
# b = a.count_fanout("./playground/buffer/_buf_0.v")
# min_cost_path, min_cost = a.run_gate_sizing("./playground/buffer/_buf_0.v")
# print(min_cost_path, min_cost)
# filename = "./playground/2/18_abc_mapped.v"
# filename = "./playground/design2-cost_estimator_1/abc_ga/init_abc_ga/netlist_1.v"
filename = "temp7.v"
library.replace_dummy(filename)
print(cost_interface.get_cost(filename))
# print(a.get_cost("./playground/1/9_abc_mapped.v"))