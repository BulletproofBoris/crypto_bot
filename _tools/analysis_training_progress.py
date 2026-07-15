import csv
import sys

path = r"C:/Users/Restorator/Documents/trader_test/trader_test/data/processed/2000_2026_1d/rl_env/training_progress.csv"

rows = []
with open(path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Convert numeric fields where possible
        for key in ['Iteration', 'Train_Return', 'Test_Return', 'Train_Sharpe']:
            if row.get(key) is not None and row[key] != '':
                try:
                    row[key] = float(row[key])
                except ValueError:
                    row[key] = None
        rows.append(row)

print(f"Total rows: {len(rows)}")
# Compute stats for Train_Return and Test_Return
train_vals = [r['Train_Return'] for r in rows if isinstance(r['Train_Return'], float)]
test_vals = [r['Test_Return'] for r in rows if isinstance(r['Test_Return'], float) and r['Test_Return'] != '']

if train_vals:
    print("\nTrain_Return stats:")
    print(f"  min: {min(train_vals)}")
    print(f"  max: {max(train_vals)}")
    print(f"  mean: {sum(train_vals)/len(train_vals):.2f}")
else:
    print("\nNo Train_Return data.")

if test_vals:
    print("\nTest_Return stats:")
    print(f"  min: {min(test_vals)}")
    print(f"  max: {max(test_vals)}")
    print(f"  mean: {sum(test_vals)/len(test_vals):.2f}")
else:
    print("\nNo Test_Return data.")

# Find best rows
if train_vals:
    best_train = max(rows, key=lambda r: r['Train_Return'] if isinstance(r['Train_Return'], float) else -float('inf'))
    print("\nBest Train_Return row:")
    print(best_train)

if test_vals:
    best_test = max(rows, key=lambda r: r['Test_Return'] if isinstance(r['Test_Return'], float) else -float('inf'))
    print("\nBest Test_Return row:")
    print(best_test)
