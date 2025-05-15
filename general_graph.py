# %%
from math import *
import numpy as np
import networkx
import casadi as cs
import plotly.graph_objects as go

# Set up graph
G = networkx.Graph()

"""
Example:
0       4
 \     /
  2 - 3 
 /     \
1       5
"""
G.add_edge(0, 2, length=1)
G.add_edge(1, 2, length=1)
G.add_edge(2, 3, length=1)
G.add_edge(3, 4, length=1)
G.add_edge(3, 5, length=1)

# %%

# Lay out the graph
pos = networkx.kamada_kawai_layout(G)

def H(*x):
    """
    The entropy function.
    """
    return sum(map(lambda x_i: x_i*cs.log(1/x_i), x))

def find_optimal_H_for_y(y, v0, v1, length = 1):
    """
    Maximizes the entropy function at the point that is `y` units from `v0` along the edge from `v0` to `v1` (the edge has total length `length`).
    """
    # y is the distance from v1 in the direction v0->v1
    opti = cs.Opti()

    xs = [opti.variable() for _ in range(len(G))]

    for x_i in xs:
        opti.set_initial(x_i, 1/len(xs))
    
    distances = [0] * len(G)
    
    # Finding the shortest path at every `y` that we evaluate allows us to handle arbitrary graphs
    for idx, node in enumerate(G.nodes):
        # If the shortest path to the other node is in the direction of v0, then negative
        # If the shortest path to the other node is in the direction of v1, then positive
        backward_distance = (networkx.shortest_path_length(G, node, v0) if networkx.has_path(G, node, v0) else float("inf"))
        forward_distance = (networkx.shortest_path_length(G, v1, node) if networkx.has_path(G, v1, node) else float("inf")) + length

        # Add y to the distances when comparing, so we're finding the shortest path to the point that is y along v0->v1
        if (forward_distance - y) < (backward_distance + y):
            distance = forward_distance
        else:
            distance = -backward_distance
        distances[idx] = distance
    
    # Maximize the entropy function
    opti.minimize(-H(*xs))
    opti.subject_to(sum(xs)==1)
    opti.subject_to(y==sum(map(lambda i: xs[i] * distances[i], range(len(xs)))))

    opti.solver('ipopt', {'ipopt.print_level':0, 'print_time':0, 'ipopt.sb':'yes'})

    sol = opti.solve()

    return H(*[sol.value(x_i) for x_i in xs])

def ratefunction(y, v0, v1, length = 1):
    """
    Calculates the ratefunction's value at the point that is `y` units from `v0` along the edge from `v0` to `v1` (the edge has total length `length`).
    """
    return -find_optimal_H_for_y(y, v0, v1, length) + log(len(G))

# %%

# This is the number of points that will be evaluated
points = 250

xs = np.array([])
ys = np.array([])
zs = np.array([])

for edge in G.edges:
    v0 = pos[edge[0]]
    v1 = pos[edge[1]]
    length = G.get_edge_data(*edge)["length"]

    for i in range(points):
        try:
            x = (i / points) * length
            start = edge[0]
            end = edge[1]
            zs = np.append(zs, [ratefunction(x, start, end, length)])
            xs = np.append(xs, [v0[0] + x*(v1[0] - v0[0])])
            ys = np.append(ys, [v0[1] + x*(v1[1] - v0[1])])
        except ValueError:
            pass
        except RuntimeError:
            pass

# %%

# Draw the graph
blank = np.array([0 for _ in range(len(zs))])
floor_xs = np.copy(xs)
floor_ys = np.copy(ys)
floor_zs = blank

# %%

# Visualize the ratefunction
fig = go.Figure(data=[
    go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='markers',
        marker=dict(
            size=2,
            color=zs,
            colorscale="Plasma",
            opacity=1.0
        ),
        name='ratefunction'
    ),
    go.Scatter3d(
        x=floor_xs,
        y=floor_ys,
        z=floor_zs,
        mode='markers',
        marker=dict(
            size=1,
            color="black",
            opacity=0.5
        ),
        name='floor'
    )])
fig.layout.scene.camera.projection.type = "orthographic"
fig.show()