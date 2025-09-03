import pandas as pd
import numpy as np

def cover(cost_matrix, rows, cols, solution):

    # print(f'rows: {rows}, cols: {cols}')

    dim = cost_matrix.shape[0]
    
    for row_idx in range(0, dim):
        # make sure this row hasn't already been covered
        if row_idx in rows: continue

        uncovered_zero_cols = set()
        for col_idx in range(0, dim):
            # make sure this column hasn't already been covered
            if col_idx in cols: continue
            if cost_matrix[row_idx, col_idx] != 0: continue

            uncovered_zero_cols.add(col_idx)

        if len(uncovered_zero_cols) != 1: continue

        # We have found a row that has exactly one un-covered zero
        # Cancel the column that it is in
        col_idx = uncovered_zero_cols.pop()
        cols.add(col_idx)
        solution.append([row_idx, col_idx])

        return cover(cost_matrix, rows, cols, solution)

    for col_idx in range(0, dim):
        # make sure this column hasn't already been covered
        if col_idx in cols: continue

        uncovered_zero_rows = set()
        for row_idx in range(0, dim):
            # make sure this row hasn't already been covered
            if row_idx in rows: continue
            if cost_matrix[row_idx, col_idx] != 0: continue

            uncovered_zero_rows.add(row_idx)

        if len(uncovered_zero_rows) != 1: continue

        row_idx = uncovered_zero_rows.pop()
        rows.add(row_idx)
        solution.append([row_idx, col_idx])

        return cover(cost_matrix, rows, cols, solution)

    return (rows, cols, solution)
    

def hungarian(cost_matrix, iter=0):
    # print('--- STARTING ---')
    # Make the matrix square 
    # If more "positions" are needed, then create a new column of zeros
    # If more "doctors" are needed, then make new rows of the max value
    [nrow, ncol] = cost_matrix.shape
    while nrow < ncol:
        cost_matrix = np.vstack((cost_matrix, (1+np.max(cost_matrix)) * np.ones(ncol)))
        nrow += 1
    while ncol < nrow:
        # Need more positions
        cost_matrix = np.column_stack((cost_matrix, np.zeros(nrow)))
        ncol += 1

    # print(cost_matrix)

    # Reduce along columns
    min_along_cols = cost_matrix.min(axis=0)
    cost_matrix = cost_matrix - min_along_cols
    
    # Reduce along rows
    min_along_rows = cost_matrix.min(axis=1)
    cost_matrix = cost_matrix - min_along_rows[:, np.newaxis]
    
    # print(cost_matrix)

    # print('Starting to cover')
    (cancelled_rows, cancelled_cols, solution) = cover(cost_matrix, set(), set(), [])

    n_cancels = len(cancelled_cols) + len(cancelled_rows)
    if n_cancels == nrow:
        return solution
    
    # Find the smallest un-cancelled element
    smallest = np.max(cost_matrix)
    for row_idx in range(0, nrow):
        if row_idx in cancelled_rows: continue
        for col_idx in range(0, ncol):
            if col_idx in cancelled_cols: continue
            smallest = min(smallest, cost_matrix[row_idx, col_idx])

    # print('Cancels')
    # print(cancelled_rows)
    # print(cancelled_cols)

    for row_idx in range(0, nrow):
        for col_idx in range(0, ncol):
            if (row_idx in cancelled_rows) and (col_idx in cancelled_cols):
                # print(f'Intersection at ({row_idx}, {col_idx})')
                cost_matrix[row_idx, col_idx] += smallest
            elif (row_idx not in cancelled_rows) and (col_idx not in cancelled_cols):
                cost_matrix[row_idx, col_idx] -= smallest
            # else:
            #     print(f'Skipping cell ({row_idx}, {col_idx})')

    # print(cost_matrix)

    return hungarian(cost_matrix, iter+1)


if __name__ == "__main__":
    sample_costs = np.array([
        [8, 4, 7],
        [5, 2, 3],
        [9, 6, 7],
        [9, 4, 8]
    ])

    print(hungarian(sample_costs))