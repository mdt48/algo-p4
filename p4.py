
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

import copy

class TreeNode:
    def __init__(self, id, data=None) -> None:
        self.id = id
        self.children = []
        if data is not None:
            self.S = data

            self.M = np.mean(data, axis=0)

            self.r = None
            m = float("-inf")

            for s in self.S:
                dist = np.sqrt( (s[0] - self.M[0])**2 + (s[1]**2 - self.M[1])**2)
                if dist > m:
                    m = dist
            self.r = dist
        else:
            self.S = []
            self.r = None
            self.M = None

    def add_child(self, node):
        self.children.append(node)

    def __eq__(self, other):
        return self.id == other.id

class knnTree:
    def __init__(self, l) -> None:
        self.l = 3
        self.tree_height = 0
        self.root = TreeNode(0)
        self.total_nodes = 1

    
    def build_tree(self, data, root=None, level=0):
        if level == self.l:
            return

        # cluster data in l parts
        kmeans = KMeans(n_clusters=self.l, random_state=0).fit(data)

        # recurse left, middle, right
        for i in range(self.l):
            cluster_indices = self._cluster_indices(i, kmeans.labels_)

            cluster_data = data[cluster_indices]
            new_node = TreeNode(self.total_nodes, cluster_data)
            self.total_nodes += 1
            if not root:
                self.root.add_child(new_node)
                self.build_tree(cluster_data, new_node, level+1)
            else:
                root.add_child(new_node)
                self.build_tree(cluster_data, new_node, level+1)
            
    def print_tree(self, level=0):
        if level == 0:
            root = self.root
            level = 0
        
        q = []
        seen = []
        q.append(root)
        self.print_rec(q, seen)

    def print_rec(self, q, seen) :
        
        if len(q) == 0:
            return
        v = q.pop(0)
        print('-'*5, end='')
        print('NODE # {}'.format(v.id))
        print('N = {}'.format(len(v.S)))
        print('r = {}'.format(v.r))
        print('-'*5, end='\n\n')
        for u in v.children:
            if not u in seen:
                seen.append(u)
                q.append(u)
        self.print_rec(q, seen)
            

    def _cluster_indices(self, clustNum, labels_array): #numpy 
        return np.where(labels_array == clustNum)[0]


def BBknn(tree, target):
    # Step 0 - initializaiton

    B = float("inf")
    L = 1
    CURRENT_NODE = tree.root

    nn = None
    nn_coord = None
    nn_dist = None


    active = defaultdict(list)
    active_means = defaultdict(list)
    active_dists = defaultdict(list)
    active_r = defaultdict(list)
    active_nodes = defaultdict(list)

    
    while True:
        print('Level {}'.format(L))
        # Step 1
        for child in CURRENT_NODE.children:
            active[L].extend(list(child.S))
            active_means[L].extend([child.M]*len(child.S))

            active_r[L].extend([child.r]*len(child.S))
            active_nodes[L].extend([child]*len(child.S))
            active_dists[L].extend( [np.sqrt ( (child.M[0] - target[0])**2 + (child.M[1] - target[1])**2 )] * len(child.S) )

        active_list_copy = copy.deepcopy(active[L])
        active_dist_copy = copy.deepcopy(active_dists[L])


        for idx, a in enumerate(active_list_copy):
            dist = active_dist_copy[idx]

            # step 2 - apply rule 1
            if dist > B + active_r[L][idx]:
                active[L].pop(idx)
                active_nodes.pop(idx)
                active_r[L].pop(idx)
                active_means[L].pop(idx)
                active_dists.pop(idx)



        # active_dists_copy = copy.deepcopy(active_dists[L])
        
        # Step 3 -- any active left at level?
        if len(active[L]) > 0:
            # Step 4
            closest_idx = np.argmin(active_dists[L])
            # closest = active[L][closest_idx]
            CURRENT_NODE = active_nodes[L][closest_idx]
            active[L].pop(closest_idx)
            active_dists[L].pop(closest_idx)
            active_r[L].pop(closest_idx)
            active_means[L].pop(closest_idx)

            if L != tree.l:
                L += 1
            else:
                # Step 5 -- rule 2
                for point in CURRENT_NODE.S:
                    dist = np.sqrt ( (CURRENT_NODE.M[0] - target[0])**2 + (CURRENT_NODE.M[1] - target[1])**2 )

                    dist_xi = np.sqrt ( (CURRENT_NODE.M[0] - point[0])**2 + (CURRENT_NODE.M[1] - point[1])**2 )
                    if dist > dist_xi + B:
                        pass
                    else:
                        d_x_xi = np.sqrt ( (target[0] - point[0])**2 + (target[1] - point[1])**2 )
                        if d_x_xi < B:
                            nn = CURRENT_NODE.id
                            nn_coord = point
                            nn_dist = d_x_xi
        else:
            L -= 1

            if L == 0:
                break
    print("closest = {}".format((nn, nn_dist, nn_coord)))



        





# %%
# Generate Points
n = 100
x = np.random.randint(0, 1000, n)
y = np.random.randint(0, 1000, n)
plt.scatter(x, y)


# %%
# Preprocessing (Tree making)
l = 3
data = np.array(list(zip(x, y)))

tree = knnTree(l)
tree.build_tree(data)

# %%
tree.print_tree()

x = np.random.randint(0, 1000, 1)
y = np.random.randint(0, 1000, 1)
target = np.array(list(zip(x, y)))

BBknn(tree, target[0])





