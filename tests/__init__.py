# # SPDX-FileCopyrightText: 2025-present 
# #
# # SPDX-License-Identifier: MIT
import pandas as pd
from scipy.optimize import linear_sum_assignment

df = pd.read_csv('D:\\JHU\\BDD\\Rank Choice Matching\\rank-order-assignment\\tests\\p1ds.csv', header=0, index_col=0)
print(df.head())
print(df.columns)

num_positions = len(df.columns) 
print(num_positions)

# # Add rows until we have `num_positions` count

# Figure out counts
name_col = df.columns[0]                      # first column is the doctor name/id
num_positions = df.shape[1]                 # exclude name column. 
num_doctors = df.shape[0]   
to_add = max(0, num_positions - num_doctors)

print(f"positions: {num_positions}, Doctors: {num_doctors}, Need to add: {to_add}")

if to_add > 0:
    # Build a DataFrame of dummy doctors with all preferences = 21
    dummy_names = [f"Dummy Doctor {i+1+num_doctors}" for i in range(to_add)]
    pref_cols = df.columns
    dummy_block = pd.DataFrame(
        num_positions+1,
        index=dummy_names,
        columns=pref_cols
    )

    print(dummy_block.head())

    # Append and reset index (optional)
    # df = pd.concat([df, dummy_block], ignore_index=False)

# (Optional) verify the shape is correct now
# assert df.shape[0] == num_positions, "Row count should now equal number of positions."

row_ind, col_ind = linear_sum_assignment(df.values)
print(row_ind)
print(col_ind)

print(df.values[row_ind, col_ind].sum())


# print(df)

# (Optional) save back out
# df.to_csv('D:\\JHU\\BDD\\Rank Choice Matching\\rank-order-assignment\\tests\\p1ds_with_dummies.csv', index=False)
