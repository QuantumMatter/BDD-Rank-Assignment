import numpy as np
from collections import deque

def adjacency_from_edges(nrow, ncol, edges):
    adj = np.zeros((nrow, ncol))
    for (i,j) in edges:
        adj[i,j] = 1
    return adj

# This is done by starting at an unmatched node in S, and trying to
# find a node in T that is unmatched, via a breadth first search
# We continue queuing nodes from those reachable from T if T is
# already in the tree
def find_augmenting_path(adj, matching):
    parent = {}

    # First, find an unmatched node to start the search from
    # Matches are edges that start in T and go to S, so they
    # will be elements in our `tree` that have a value -1
    unmatched_rows_mask = np.all(adj == 0, axis=1)
    unmatched_rows = np.where(unmatched_rows_mask)[0]

    queue = deque(unmatched_rows)
    visited = set(unmatched_rows)

    while len(queue) > 0:
        row = queue.pop()

        neighbor_columns = np.where(adj[row,:] == 1)[0]
        for col in neighbor_columns:
            if matching[col] is not None:
                # The node has already been matched
                # Queue its matched node in S
                queue.append(matching[col])
                visited.add(matching[col])
                parent[matching[col]] = row
            else:
                # The node has not been matched, so we've
                # found an augmenting path!
                # Reconstruct the path, and return it to the caller

                # Add the current edge, from S to T
                # Then, find the edge that went from T to S prior
                path = [(row, col)]
                path_row = row
                while path_row != None:
                    # Find the edge that goes from T to path_row in S



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

    ## Keep track of the elements of our solution
    # Instantiate our potential
    # y_row = np.zeros(nrow)
    # y_col = np.zeros(ncol)
    y_row = cost_matrix.min(axis=1)
    y_col = np.zeros(ncol)
    # Instantiate the matching
    # Essentially a collection of edges from T to S,
    # where col is the ID of an elemnt in T, and its
    # value will the be ID of the element its matched to in S
    matching = { col: None for col in range(ncol) }

    # While we don't have a complete matching, try to iterate the algorithm
    while any(map(lambda value: value is None, matching.values)):
        # First, try to find an augmenting path
        augmenting_path = find_augmenting_path()

        if augmenting_path is not None:
            # We need to invert the matchings along the new path
            pass
        else:
            # We need to update the dual
            pass