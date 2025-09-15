def test_unit_1():
    assert True

def test_import():
    from rank_order_assignment import dual_hungarian
    assert dual_hungarian is not None

def test_hungarian_def(): 
    from rank_order_assignment.dual_hungarian import hungarian
    assert hungarian is not None

def test_more_doctors_than_position(): 
    import numpy as np
    from rank_order_assignment.dual_hungarian import hungarian
    from scipy.optimize import linear_sum_assignment

    table = np.array([
    [3, 7, 1, 9, 5],
    [8, 2, 4, 6, 0],
    [10, 1, 7, 3, 9]
    ], dtype=int)

    our_assignments, _ = hungarian(table)
    _, scipy_assignments= linear_sum_assignment(table)

    position_encoded_list = []
    pos = 0

    for assignment in our_assignments: 
        position_encoded_list.append((assignment, pos))
        pos += 1

    sorted_assignment = sorted(position_encoded_list, key=lambda item: item[0])

    our_processed_assignments = []
    for assignment in sorted_assignment: 
        our_processed_assignments.append(assignment[1])

    print(our_processed_assignments[:len(table)])
    print(scipy_assignments)  
    print("------------------------------------------------------------------------------------------------------")

def test_with_same_ratings(): 
    import numpy as np
    from rank_order_assignment.dual_hungarian import hungarian
    from scipy.optimize import linear_sum_assignment

    table = np.array([
    [3, 3, 7, 1],
    [5, 2, 2, 8],
    [9, 4, 9, 6],
    [0, 0, 10, 7]
    ], dtype=int)

    our_assignments, _ = hungarian(table)
    _, scipy_assignments= linear_sum_assignment(table)

    position_encoded_list = []
    pos = 0

    for assignment in our_assignments: 
        position_encoded_list.append((assignment, pos))
        pos += 1

    sorted_assignment = sorted(position_encoded_list, key=lambda item: item[0])

    our_processed_assignments = []
    for assignment in sorted_assignment: 
        our_processed_assignments.append(assignment[1])

    print(our_processed_assignments[:len(table)])
    print(scipy_assignments) 
    print("------------------------------------------------------------------------------------------------------")


def test_with_same_ratings_and_more_doctors_than_position(): 
    import numpy as np
    from rank_order_assignment.dual_hungarian import hungarian
    from scipy.optimize import linear_sum_assignment

    table = np.array([
    [3, 7, 5, 9, 5],
    [8, 2, 2, 2, 0],
    [10, 1, 7, 3, 9]
    ], dtype=int)

    our_assignments, _ = hungarian(table)
    _, scipy_assignments= linear_sum_assignment(table)

    position_encoded_list = []
    pos = 0

    for assignment in our_assignments: 
        position_encoded_list.append((assignment, pos))
        pos += 1

    sorted_assignment = sorted(position_encoded_list, key=lambda item: item[0])

    our_processed_assignments = []
    for assignment in sorted_assignment: 
        our_processed_assignments.append(assignment[1])

    print(our_processed_assignments[:len(table)])
    print(scipy_assignments) 
    print("------------------------------------------------------------------------------------------------------")
   
    # print(our_assignments)
    # print(scipy_assignments)

# def test_with_unranked(): 
#     import numpy as np
#     from rank_order_assignment.dual_hungarian import hungarian
#     from scipy.optimize import linear_sum_assignment

#     table = np.array([
#     [1, 2, np.nan],
#     [4, np.nan, 6],
#     [7, 8, np.nan],
#     [np.nan, 1, 3],
#     [5, np.nan, 9]
#     ])

#     our_assignments = hungarian(table)
#     scipy_assignments = linear_sum_assignment(table)

if __name__=="__main__": 
    test_more_doctors_than_position()
    test_with_same_ratings()
    test_with_same_ratings_and_more_doctors_than_position()
