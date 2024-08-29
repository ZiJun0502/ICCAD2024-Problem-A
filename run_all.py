import subprocess
import csv
import time
import os
from utils.cost_interface import CostInterface

# designs = [f"design{i}.v" for i in range(1, 7)]
# cost_estimators = [f"cost_estimator_{i}" for i in range(1, 9)]
designs = [f"design{i}.v" for i in [5,6]]
cost_estimators = [f"cost_estimator_{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8]]
# designs = [f"design{i}.v" for i in range(1, 7)]
# cost_estimators = [f"cost_estimator_{i}" for i in range(1, 9)]
output_csv = "results.csv"

# List to store results
results = []

for design in designs:
    for cost_estimator in cost_estimators:
        output_dir = os.path.join("./playground", f"{design[:-2]}-{cost_estimator}")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        cost_estimator_path = f"./release/cost_estimators/{cost_estimator}"
        library_path = "./release/lib/lib1.json"
        netlist_path = f"./release/netlists/{design}"
        output_path = os.path.join(output_dir, "a.v")
        command = [
            "python", "main.py",
            "-cost_function", cost_estimator_path,
            "-library", library_path,
            "-netlist", netlist_path,
            "-output", output_path
        ]
        cost_interface = CostInterface(
            cost_estimator_path=cost_estimator_path,
            library_path=library_path
        )
        try:
            # Run the command and capture the output
            print(f"Running design {design} and cost estimator {cost_estimator}")
            
            start_time = time.time()
            proc = subprocess.check_output(command, stderr=subprocess.STDOUT)
            end_time = time.time()

            output = proc.decode('utf-8')
            
            # Extract min_cost from the captured output
            min_cost = None
            try:
                min_cost = cost_interface.get_cost(output_path)
            except:
                min_cost = None
            # output_lines = output.split('\n')
            # for line in output_lines:
            #     if 'FINAL_min_cost' in line:
            #         min_cost = float(line.split()[-1])
            #         break
            
            if min_cost is None:
                print(f"min_cost not found for design {design} and cost estimator {cost_estimator}")
                continue

            # Save the results
            print(f"\tmin cost: {min_cost:.6f}, time elapsed: {end_time - start_time:.2f}")
            print("\tDateTime:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))

            results.append([design, cost_estimator, f"{min_cost:.6f}", f"{end_time - start_time:.2f}"])

        except subprocess.CalledProcessError as e:
            print(f"Error running command for design {design} and cost estimator {cost_estimator}")
            continue

# Write results to CSV
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Design", "Cost Estimator", "Min Cost", "Duration (s)"])
    writer.writerows(results)

print(f"Results saved to {output_csv}")
