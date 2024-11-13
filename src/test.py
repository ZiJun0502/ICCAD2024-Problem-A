from CostInterface import CostInterface

a = CostInterface("./release/cost_estimators/cost_estimator_1", "./release/lib/lib1.json")
print(a.get_cost("temp_mapped.v"))
# print(a.get_cost("./playground/1/9_abc_mapped.v"))