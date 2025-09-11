# Hungarian Algorithm

The Hungarian algorithm is used to solve minimum cost assignment problems. Consider n locations, and each location needs to be serviced by one worker. Each worker is at a different location, and therefore will have a different travel time to each of the locations. The algorithm would determine which workers should go to which locations to minimize the time spent travelling.

The the inventors of the algorthm use a combination of graph theory and linear programming to derive the process and prove correctness. The problem is first modelled by a linear program. The primal is a bipartite matching problem, and the dual tracks intermediate values of each of the nodes. With this formulation, we can use the theorm that states any solution that satisfies both the primal and the dual is an optimal solution. For a solution to satisfy both the primal and the dual, it must conform to the complementary slack constraints. In this problem, those limit the edges that can be considered for the matching. Therefore, we start by trying to find the matching. When we get stuck, we update the dual but preserve our complementary slack constraints. The updated dual should have exposed a new edge for our graph, so we restart the cycle. After each loop, the matching grows by one edge. Thus after we iterate enough times, we will have a solution the primal (and therefore is a perfect matching) and the dual, and therefore optimal.

# Linear Programming

Linear programming describes the process of solving a linear optimization problem given an additional set of linear constraints. It's a classic convex optimization problem.

## Standard Form

The function to be optimized is called the objective, and the constraints are organized into three categories
1. Inequality constraints (<=, <, >, >=)
2. Equality Constraints (=)
3. Non-Negative Constraints (>= 0)
4. Unconstrained

Because we're dealing with linear systems, all of these can be represented as matrices and vectors

### Objective Function

$\text{max } c^Tx$

* `c` - The function to be minimized
* `x` - The optimal solution to be found

### Constraints

Let `i` be the `i`th constraint of our system. Depending on what type of constraint it is, it'll be in one of these forms.

* Inequality: $a_i^Tx \leq b_i$
  * Note that $\geq$ can be reduced to $\leq$ by simply switching the sign of $a_i$ and  $b_i$
  * The case of a strict in equality is a little harder, and something we won't consider
* Equality: $a_i^Tx = b_i$
* Non-Negative: $a_i^T \geq 0$
* Unconstrained: $\bar{0} = \bar{0}$

Let's say that the first `k` terms are equality constraints, and the last `n-k` constraints are inequalities.

## General Properties

Note that each constraint defines a hyper-plane, and thefore the set of all constraints defines a region. This is called the feasibility region.
* If the feasibility region is empty, there is no solution
* If it is unbounded, there exists an objective for which no solution exists
* If it is bounded, there exists a solution for every objective

### Farkas' Lemma

We can show that an LP problem is infeasible there exists a combination of constraints for which there is no solution.

$$
Ax \leq b \\
y^TAx \leq y^Tb \\
y^TA = 0 \rightarrow 0 \leq y^Tb
$$
So if we can find a $y$ such that $y^TA$ is 0, but $y^Tb$ is negative, then the system is infeasible.

## The Dual

The concept of the dual flips the notion of Farkas' Lemma. If we know that the $y^Tb < 0$  is an infeasible solution, lets maximize $y^Tb$ - this will definitely put us in the feasible region. To make this useful, we want to find the lowest value of $y^Tb$ that is still an upper bound. This, we have another optimization problem, and it can be shown that is another linear programming problem.

$$
\begin{alignedat}{2}
& \text{Primal} &\qquad\qquad\qquad\qquad\qquad& \text{Dual} \\
& \text{max } c^Tx && \text{min } y^Tb \\
\forall i \in [k], \qquad & A_i^{(r)}x \leq b_i & \forall j \in [p], \qquad & y^TA_j^{(c)} \geq c_j \\
\forall i \in (k, m], \qquad & A_i^{(r)}x = b_i & \forall i \in (p, n], \qquad &y^TA_j^{(c)} = c_j \\
\forall j \in (m, p], \qquad & x_j \geq 0 & \forall i \in (n, k], \qquad & y_j \geq 0\\ 
\end{alignedat}
$$

Essentially, the constraint matrix gets transposed and is now operating on the primal objective function.

### Weak Duality

If a primal solution is a lower bound, and a dual solution is an upper bound, it can be said that they apply this to each other too.

The primal is a lower bound for the dual. \
The dual is an uppwer bound for the primal.

### Strong Duality

If the primal has an optimal solution, and the dual has an optimal solution, they are the same point.

$$ \text{max } c^Tx = \text{min } y^Tb $$

### Complementary Slackness

If we assume that our problem as strong duality, then we derive some additional constrains for our problem.

## Integer Properties

We can't express integer constraints to in our LP constraints, because they're not linear. For example, we can't say that we want to only consider $x$ such that each of its elements is an integer. This is undesirable when the solution needs to be interpretted in a discretized world, for example in the distribution of goods.

However, the Birkhoff theorm says that this doesn't actually matter. The integer constraint can be reduced to a non-negative constraint. Then, any solution to this LP will also have an integer solution.

The Hungarian algorithm is actually a proof of this theorm. From the process we construct below, we know that we'll arrive at a solution that is integer valued. With some additional work, we could show how to convert a continously valued solution into an integer valued one, but we won't do that here.

# Algorithm

We start by formulating the assignment problem as a linear programming problem.

## LP Formulation

We choose the primal to be the general assignment problem. It's objective is to minimize the cost of all of the assignments.

$\text{min }c^Tx = \text{min } \sum_E{c_ex_e}$
* `c_e` - The cost of the edge between two nodes
* `x_e` - 1 if the edge is used, 0 if not

Our constraints are used to implement the matching behavior. Specifically, each worker should only be assigned to one job. Each worker node should only have one edge, which has the value `x_e=1`

$\forall u \in U, \displaystyle\sum_{v\in N(u)}{x_{u,v}}=1$

We could also write an integral constraint, but as we've already discussed, this will be relaxed to a non-negative constraint, so we'll write that instead.

$\forall e \in E, x_e \geq 0$

Thus, we end up with a constraint matrix with the properties that

* Each row will have a 1 for everything in the neighborhood of that vertex
  * Let's say that row `i` is the constraint for the `i`th vertex. Each column represents another vertex. Therefore, each column in this row will the `1` if the vertex `v_i` has an edge to the vertex `v_j`.
  * This is called the edge incidence vector
* Each column only two nonzero elements, and each are equal to one. These are the start and end of each of the edges in our graph.

Now, let's consider the dual. The object will be $\max{y^T }b$, since our primal is a maximization. Its constraint matrix will the the transpose of the primal's. This leads us to the properties

* Each row has only two non-zero elements, and representing the edges of the graph. These are become inequality constraints, such that $y_i + y_j \leq b_{i,j}$

If we consider the strong duality of the problem, we find that the complementary slackness constraint is such that
$y_i + y_b = b_{i,j}$

## Overview

From our LP Formulation, we have the requirements that
* Primal
  * The final solution is a perfect matching in a bipartite tree
* Dual
  * We must maintain that $y_i + y_j < c_{(i,j)}$
* Complementary Slackness
  * In our final solution, $y_i + y_j < c_{(i,j)}$

We'll choose the dual and complementary slackness constraints to be invariants throughout our algorithm.

## Primal Step

We'll start by trying to establish a perfect matching in the barptite tree. However, we'll only consider edges that satisfy the complementary slackness requirement. If such a matching is found, then all constraints of our LP problem are satisfied, and we know we've arrived at the optimal solution.

We'll perform the bipartite matching by building an alternating tree. The matching will be found if the tree covers the entire graph.

## Dual Step

If the primal step does not find a solution, we'll need to update the dual.

To do this, we'll try to expand the number of edges that can be considered in the primal step. To do this, we need to add at least one more edge that is not currently in the set.

We'll pick the node in the unmatched set of nodes that has the smallest weight to get to a node in the matched set. To make the edge statisfy the constraint $y_i + y_j = c_{(i,j)}$, we'll need to adjust $y_i$ and/or $y_j$. We'll choose to update the node that is already in the matched set, so we preserve the relationship of all of the nodes not in the set. This will be the value $\delta = c_{(i,j)} - y_i - y_j$. However, this update will effect its constraint. What we find is that we need to all nodes in the bipartite group with $y_i$ need change by $\delta$, and all nodes in the other group need to change be $-\delta$.

After this update to $y$ there should be another edge available for the matching, so we loop back to the primal step.