import subprocess
import csv
import time

designs = [f"design{i}.v" for i in range(1, 7)]
cost_estimators = [f"cost_estimator_{i}" for i in range(1, 9)]
# designs = [f"design{i}.v" for i in range(1, 7)]
# cost_estimators = [f"cost_estimator_{i}" for i in range(1, 9)]
output_csv = "results.csv"

# List to store results
results = []

for design in designs:
    for cost_estimator in cost_estimators:
        command = [
            "python", "main.py",
            "-cost_function", f"./release/cost_estimators/{cost_estimator}",
            "-library", "./release/lib/lib1.json",
            "-netlist", f"./release/netlists/{design}",
            "-output", "./a.v"
        ]
        
        try:
            # Run the command and capture the output
            print(f"Running design {design} and cost estimator {cost_estimator}")
            
            start_time = time.time()
            proc = subprocess.check_output(command, stderr=subprocess.STDOUT)
            end_time = time.time()

            output = proc.decode('utf-8')
            
            # Extract min_cost from the captured output
            min_cost = None
            output_lines = output.split('\n')
            for line in output_lines:
                if 'FINAL_min_cost' in line:
                    min_cost = float(line.split()[-1])
                    break
            
            if min_cost is None:
                print(f"min_cost not found for design {design} and cost estimator {cost_estimator}")
                continue

            # Save the results
            print(f"\tmin cost: {min_cost:.6f}, time elapsed: {end_time - start_time:.2f}")
            print("DateTime:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end)))

            results.append([design, cost_estimator, f"{min_cost:.6f}", f"{end_time - start_time:.2f}"])

        except subprocess.CalledProcessError as e:
            print(f"Error running command for design {design} and cost estimator {cost_estimator}: {e.output.decode('utf-8')}")
            continue

# Write results to CSV
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Design", "Cost Estimator", "Min Cost", "Duration (s)"])
    writer.writerows(results)

print(f"Results saved to {output_csv}")
