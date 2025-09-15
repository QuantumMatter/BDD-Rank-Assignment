import numpy as np
from collections import deque

# This is done by starting at an unmatched node in S, and trying to
# find a node in T that is unmatched, via a breadth first search
# We continue queuing nodes from those reachable from T if T is
# already in the tree
def find_augmenting_path(adj, matching):
    n = adj.shape[0]

    # First, find an unmatched node to start the search from
    unmatched_rows = [r for r in range(n) if r not in matching]

    parent_row = -np.ones(n, dtype=int)
    parent_col = -np.ones(n, dtype=int)
    visited_rows = np.zeros(n, dtype=bool)
    visited_cols = np.zeros(n, dtype=bool)
    queue = deque(unmatched_rows)

    for row in unmatched_rows:
        visited_rows[row] = True

    augmenting_col = -1

    while len(queue) > 0:
        row = queue.popleft()
        # print(f'Exploring row { row }')

        neighbor_columns = np.where(adj[row,:] == 1)[0]
        for col in neighbor_columns:
            if visited_cols[col]: continue

            visited_cols[col] = True
            parent_row[col] = row

            if matching[col] != -1:
                # The node has already been matched
                # Queue its matched node in S
                matched_row = matching[col]
                # print(f'Row { row } has neighbor column { col } that is already matched to { matched_row }')
                queue.append(matched_row)
                visited_rows[matched_row] = True
                parent_col[matched_row] = col
            else:
                # print(f'Row { row } has an augmenting path to column { col }')
                augmenting_col = col
                break
                
        if augmenting_col != -1:
            break

    if augmenting_col == -1:
        return None, visited_rows, visited_cols
    

    # The node has not been matched, so we've
    # found an augmenting path!
    # Reconstruct the path, and return it to the caller

    path = []
    curr_col = augmenting_col
    while curr_col != -1:
        curr_row = parent_row[curr_col]
        path.append((curr_row, curr_col))

        # Move to the previous row in the path
        if curr_row in unmatched_rows:
            break
        
        curr_col = parent_col[curr_row]

    path.reverse()
    return path, visited_rows, visited_cols


def hungarian(cost_matrix):
    # Make the matrix square 
    # If more "positions" are needed, then create a new column of zeros
    # If more "doctors" are needed, then make new rows of the max value
    [nrow, ncol] = cost_matrix.shape
    while nrow < ncol:
        cost_matrix = np.vstack((cost_matrix, np.full(ncol, 1e10)))
        nrow += 1
    while ncol < nrow:
        # Need more positions
        cost_matrix = np.column_stack((cost_matrix, np.full(nrow, 1e10)))
        ncol += 1

    ## Keep track of the elements of our solution
    # Instantiate our potential
    y_row = cost_matrix.min(axis=1)
    y_col = np.zeros(ncol, dtype=int)
    # Instantiate the matching
    # Essentially a collection of edges from T to S,
    # where col is the ID of an elemnt in T, and its
    # value will the be ID of the element its matched to in S
    # matching[j]=i  -->  row i is matched with column j
    matching = -np.ones(ncol, dtype=int)

    # While we don't have a complete matching, try to iterate the algorithm
    while np.any(matching == -1):
        # print(matching)
        adj = np.zeros((nrow, ncol), dtype=int)
        for i in range(nrow):
            for j in range(ncol):
                # Assume the bipartite graph is dense, and all students rank all programs
                weight = cost_matrix[i,j]
                if weight == y_row[i] + y_col[j]:
                    adj[i,j] = 1

        # First, try to find an augmenting path
        augmenting_path, visited_rows, visited_cols = find_augmenting_path(adj, matching)

        if augmenting_path is not None:
            # We need to invert the matchings along the new path
            # The path should start and end with the "unmatched" edges
            # Start traversing at the newly discovered leaf
            for (i,j) in augmenting_path:
                matching[j] = i

        else:
            # We need to update the dual
            # Find the minimum value to adjust the potential by,
            # such that another tight edge is created between S and T
            # and therefore a new augmenting path can be found
            # This edge will exist between a visited row, and unvisited column
            min_delta = 1e10
            for i in range(nrow):
                if not visited_rows[i]: continue
                for j in range(ncol):
                    if visited_cols[j]: continue
                    slack = cost_matrix[i,j] - (y_row[i] + y_col[j])
                    min_delta = min(min_delta, slack)

            for i in range(nrow):
                if not visited_rows[i]: continue
                y_row[i] += min_delta
            
            for j in range(ncol):
                if not visited_cols[j]: continue
                y_col[j] -= min_delta

    # We have our matching!
    # print(matching)
    return matching, np.sum(y_col) + np.sum(y_row)


if __name__ == '__main__':
    sample_costs = np.array([
        [8, 4, 7],
        [5, 2, 3],
        [9, 6, 7],
        [9, 4, 8]
    ], dtype=int)

    print(hungarian(sample_costs))

    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(seed=303)

    for n in range(10, 100):
        nrow = rng.integers(3, n+1)
        ncol = n - nrow

        cost = rng.integers(0, ncol, (nrow, ncol))

        if (n % 10) == 0:
            print(f"TEST STEP n={n} =======")
            print(cost)

        msol, _ = hungarian(cost)
        mcost = 0
        for (j,i) in enumerate(msol):
            if i >= nrow: continue # Padding
            if j >= ncol: continue # Padding
            mcost += cost[i,j]

        srow, scol = linear_sum_assignment(cost)
        scost = cost[srow, scol].sum()

        assert mcost == scost, "Solution doesn't match reference!"
        print(f'Passed n={n}')


# ===== Convenience utilities to build costs, extract match, and compute cost =====
from collections import defaultdict

def build_cost_matrix(preferences, capacities, large_penalty=10**6):
    """
    Expand hospitals into individual slots and build a rectangular cost matrix.
    preferences: List[List[int]] where preferences[d] is an ordered list of hospital ids for doctor d (0 = best)
    capacities:  List[int] of length num_hospitals, or a single int (uniform capacity)
    Returns:
        C (ndarray) shape (num_doctors, num_slots)
        hospital_slots: List[Tuple[int,int]] mapping slot_index -> (hospital_id, slot_id_within_hospital)
    """
    import numpy as np
    num_doctors = len(preferences)
    # infer hospital count from preferences
    max_h = -1
    for pref in preferences:
        for h in pref:
            if h > max_h:
                max_h = h
    num_hospitals = max_h + 1 if max_h >= 0 else 0

    # normalize capacities
    if isinstance(capacities, int):
        capacities = [capacities] * num_hospitals
    if len(capacities) != num_hospitals:
        raise ValueError("capacities length must equal number of hospitals inferred from preferences")

    # build slot list
    hospital_slots = []
    for h, cap in enumerate(capacities):
        for s in range(cap):
            hospital_slots.append((h, s))
    num_slots = len(hospital_slots)

    # precompute ranks for each doctor: dict hospital->rank
    ranks = []
    for pref in preferences:
        ranks.append({h: r for r, h in enumerate(pref)})

    # build rectangular cost
    C = np.full((num_doctors, num_slots), large_penalty, dtype=float)
    for d in range(num_doctors):
        for j, (h, _) in enumerate(hospital_slots):
            C[d, j] = ranks[d].get(h, large_penalty)

    return C, hospital_slots

def extract_match(matching, num_doctors, hospital_slots):
    """
    Convert 'matching' (length = num_cols) mapping column->row into
    - match_list: List[(doctor_id, hospital_id)]
    - by_hospital: Dict[hospital_id, List[doctor_id]]
    Filters out any padding matches beyond original doctors/slots.
    """
    match_list = []
    by_hospital = defaultdict(list)
    num_slots = len(hospital_slots)
    for col, row in enumerate(matching):
        if row == -1: 
            continue
        if row >= num_doctors or col >= num_slots:
            # padding introduced by the internal square padding
            continue
        h, _ = hospital_slots[col]
        match_list.append((int(row), int(h)))
        by_hospital[int(h)].append(int(row))
    match_list.sort(key=lambda x: x[0])
    # sort doctor lists per hospital for readability
    for h in list(by_hospital.keys()):
        by_hospital[h] = sorted(by_hospital[h])
    return match_list, dict(by_hospital)

def compute_total_cost(preferences, match_list, LARGE=10**6):
    """
    Compute total/average rank cost for a given assignment.
    If a doctor is assigned to a hospital not in their preference list,
    LARGE penalty is used.
    """
    ranks = [{h: idx for idx, h in enumerate(pref)} for pref in preferences]
    total = 0.0
    per_doc = {}
    for d, h in match_list:
        c = ranks[d].get(h, LARGE)
        total += c
        per_doc[d] = c
    avg = total / max(len(match_list), 1)
    return total, avg, per_doc

def assign_doctors(preferences, capacities, large_penalty=10**6):
    """
    High-level helper that builds the cost matrix, runs Hungarian,
    and returns the assignment plus summary.
    """
    import numpy as np
    C_rect, hospital_slots = build_cost_matrix(preferences, capacities, large_penalty=large_penalty)
    matching, dual_value = hungarian(C_rect.copy())  # matching: columns -> row indices
    match_list, by_hospital = extract_match(matching, C_rect.shape[0], hospital_slots)
    total, avg, per_doc = compute_total_cost(preferences, match_list, LARGE=large_penalty)
    summary = {
        "total_cost": float(total),
        "avg_cost": float(avg),
        "num_assigned": len(match_list),
        "num_doctors": len(preferences),
        "num_hospitals": len({h for h, _ in hospital_slots}),
        "dual_value": float(dual_value),
    }
    return match_list, by_hospital, summary, per_doc
