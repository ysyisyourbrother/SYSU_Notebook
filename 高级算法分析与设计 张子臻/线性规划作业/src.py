import gurobipy as gp
from gurobipy import GRB

categories, minNutrition, maxNutrition = gp.multidict({
    'calories': [2000, 2250],
    'Vitamin': [5000, 50000],})

foods, cost = gp.multidict({
    'Corn':         0.18,
    'Milk':         0.23,
    'Wheat Bread':  0.05,})

# Nutrition values for the foods
nutritionValues = {
    ('Corn', 'calories'):             72,
    ('Corn', 'Vitamin'):              107,
    ('Milk', 'calories'):             121,
    ('Milk', 'Vitamin'):              500,
    ('Wheat Bread', 'calories'):      65,
    ('Wheat Bread', 'Vitamin'):       0,}

# Model
m = gp.Model("diet")

# Create decision variables for the foods to buy
buy = m.addVars(foods, name="buy")

m.setObjective(buy.prod(cost), GRB.MINIMIZE)

m.addConstrs((gp.quicksum(nutritionValues[f, c] * buy[f] for f in foods)
             == [minNutrition[c], maxNutrition[c]]
             for c in categories), "_")

def printSolution():
    if m.status == GRB.OPTIMAL:
        print('\nCost: %g' % m.ObjVal)
        print('\nBuy:')
        for f in foods:
            if buy[f].X > 0.0001:
                print('%s %g' % (f, buy[f].X))
    else:
        print('No solution')


# Solve
m.optimize()
printSolution()

print('\nAdding constraint: at most 6 servings of dairy')
m.addConstr(buy.sum(['Corn']) <= 10, "limit_dairy")
m.addConstr(buy.sum(['Milk']) <= 10, "limit_dairy")
m.addConstr(buy.sum(['Wheat Bread']) <= 10, "limit_dairy")

# Solve
m.optimize()
printSolution()