# Copyright 2019 D-Wave Systems, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import networkx as nx
from collections import defaultdict
from itertools import combinations
from qdeepsdk import QDeepHybridSolver
import math
import numpy as np

# Set tunable parameters
num_reads = 1000
gamma = 80

# Set up our graph
G = nx.gnp_random_graph(40, 0.2)
print("Graph on {} nodes created with {} out of {} possible edges.".format(
    len(G.nodes), len(G.edges), len(G.nodes) * (len(G.nodes) - 1) / 2))

# Set up our QUBO dictionary
Q = defaultdict(int)

# Fill in Q matrix based on the graph structure
for u, v in G.edges:
    Q[(u, u)] += 1
    Q[(v, v)] += 1
    Q[(u, v)] += -2

for i in G.nodes:
    Q[(i, i)] += gamma * (1 - len(G.nodes))

for i, j in combinations(G.nodes, 2):
    Q[(i, j)] += 2 * gamma

# Convert QUBO dictionary to a numpy array.
n = len(G.nodes)
Q_array = np.zeros((n, n))
for (i, j), value in Q.items():
    Q_array[i, j] = value

# Set chain strength (if needed by your problem formulation)
chain_strength = gamma * len(G.nodes)

# Run our QUBO on the QPU using the new API.
solver = QDeepHybridSolver()
solver.token = "mtagdfsplb"
response = solver.solve(Q_array)

# Extract the best sample solution from the response.
# The response is a dictionary with key 'QdeepHybridSolver' containing a 'configuration' list.
sample = response['QdeepHybridSolver']['configuration']

# Check if the best solution found is feasible (balanced partition) and count the cut edges.
if sum(sample) in [math.floor(len(G.nodes) / 2), math.ceil(len(G.nodes) / 2)]:
    num_cut_edges = 0
    for u, v in G.edges:
        num_cut_edges += sample[u] + sample[v] - 2 * sample[u] * sample[v]
    print("Valid partition found with", num_cut_edges, "cut edges.")
else:
    print("Invalid partition.")
