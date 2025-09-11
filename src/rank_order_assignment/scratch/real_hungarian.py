import numpy as np
from collections import deque

def build_alternating_tree(edges, tree):
    [nrow, ncol] = tree.shape

    # Start by finding an unmatched node in S
    # Matches are edges that start in T and go to S, so they
    # will be elements in our `tree` that have a value -1
    unmatched_rows_mask = np.all(tree != -1, axis=1)
    unmatched_rows = np.where(unmatched_rows_mask)[0]
    for root in unmatched_rows:
        # Consider all of its neighbors via tight edges
        queue = deque()
        row_visited = np.full(nrow, False, dtype=bool)
        path_for_s = {}
        queue.append(root)

        while len(queue) > 0:
            row = queue.pop()
            row_visited[row] = True

            for (i,j) in edges:
                if i != row: continue

                # Check if the node in T (j) is already in the tree
                if np.any(tree[:,j] != 0):
                    # Add a new edge
                    tree[row, j] = 1
                    # Consider the neighbors of the node that `j` is matched with
                    j_match = np.where(tree[:,j] == -1)[0]
                    print(f'{j}: {tree[:,j]}')
                    assert len(j_match) == 1, f'T Node {j} has { len(j_match) } matches; expected 1'
                    j_match = j_match[0]
                    if row_visited[j_match] != True:
                        queue.append(j_match)
                        path_for_s[j_match] = path_for_s.get(row, []) + [(row, j), (j_match, j)]
                else:
                    # We've found an augmenting path!
                    for (pi, pj) in path_for_s.get(row, []):
                        tree[pi, pj] *= -1
                    tree[i,j] = -1

                    # Recurse back to finding a new root
                    return build_alternating_tree(edges, tree)
            
        return tree


def hungarian(cost_matrix):
    # Make the matrix square 
    # If more "positions" are needed, then create a new column of zeros
    # If more "doctors" are needed, then make new rows of the max value
    [nrow, ncol] = cost_matrix.shape
    while nrow < ncol:
        cost_matrix = np.vstack((cost_matrix, np.zeros(ncol)))
        nrow += 1
    while ncol < nrow:
        # Need more positions
        cost_matrix = np.column_stack((cost_matrix, np.zeros(nrow)))
        ncol += 1

    # Instantiate our potential
    # y_row = np.zeros(nrow)
    # y_col = np.zeros(ncol)
    y_row = cost_matrix.min(axis=1)
    y_col = np.zeros(ncol)

    # The adjacency matrix for our alternating tree
    # [i,j] == 1  -->  There's an edge from i to j
    # [i,j] == -1 -->  There's an edge from j to i
    # [i,j] == 0  -->  There's not an edge from i to j
    tree = np.zeros((nrow, ncol))

    while True:
        # Try to find as solution to the primal
        # Find the set of tight edges
        # Loop over all edges. If the sum of the potential at its nodes equals
        # its weight, then it is tight
        tight_edges = []
        for i in range(nrow):
            for j in range(ncol):
                # Assume the bipartite graph is dense, and all students rank all programs
                weight = cost_matrix[i,j]
                if weight == y_row[i] + y_col[j]:
                    tight_edges.append((i,j))

        # Continue to build an alternating tree
        tree = build_alternating_tree(tight_edges, tree)

        # If there is a matching for all nodes on the right, then we're done
        if np.all(np.any(tree == -1, axis=0)):
            # We're done!
            return tree

        # Otherwise, update the dual
        # Find the minimum amount to update the potential by, such that
        # a new edge can be made between a node in S and one note matched in T
        unmatched_cols = np.all(tree != -1, axis=0)
        unmatched_cols = np.where(unmatched_cols)[0]
        min_delta = 99999
        for unmatched_j in unmatched_cols:
            for i in range(nrow):
                min_delta = min(cost_matrix[i, unmatched_j] - y_row[i] - y_col[j])
                assert min_delta > 0
        y_row[np.any(tree != 0, axis=1)] += min_delta
        y_col[np.any(tree != 0, axis=0)] -= min_delta
        # Recurse!


if __name__ == '__main__':
    sample_costs = np.array([
        [8, 4, 7],
        [5, 2, 3],
        [9, 6, 7],
        [9, 4, 8]
    ])

    print(hungarian(sample_costs))