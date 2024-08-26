from subprocess import check_output
class CostInterface:
    _instance = None
    _initialized = False
    cost_estimator_path = ""
    library_path = ""
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CostInterface, cls).__new__(cls)
        return cls._instance
    def __init__(self, cost_estimator_path="", library_path=""):
        if not self._initialized or (cost_estimator_path != "" and library_path != ""):
            self.cost_estimator_path = cost_estimator_path
            self.library_path = library_path
            self._initialized = True
    def __str__(self):
        return self.cost_estimator_path + "," + self.library_path
    
    def get_cost(self, design_path):
        cost_output = design_path.replace('.v', '_cost.txt')

        proc = check_output([self.cost_estimator_path, \
            '-library', self.library_path, 
            '-netlist', design_path,
            '-output', cost_output
        ])
        cost = float((proc.decode('utf-8')).strip(' ').split('=')[1])
        return cost