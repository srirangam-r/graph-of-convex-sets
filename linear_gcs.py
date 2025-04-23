import numpy as np
import cvxpy as cp
from scipy.spatial import HalfspaceIntersection
from shapely.geometry import Polygon as ShapelyPolygon, Point
from scipy.optimize import linprog
import random

class GCSPlanner:
    def __init__(self, halfspace_regions, start, goal, world_bounds,
                 adj=None, solve_integer=False):
        self.regions       = halfspace_regions
        self.start         = np.array(start)
        self.goal          = np.array(goal)
        self.world_bounds  = np.array(world_bounds)
        self.solve_integer = solve_integer

        # adjacency override
        self.adj           = adj
        self._skip_overlap = (adj is not None)

    def _order(self, pts):
        c   = pts.mean(axis=0)
        ang = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
        return pts[np.argsort(ang)]

    def build_overlap_adj(self):
        n  = len(self.regions) + 2
        A  = np.zeros((n,n), int)
        s, g = n-2, n-1

        # turn each region into a Shapely polygon
        polys = [
            ShapelyPolygon(self._order(r.intersections))
            for r in self.regions
        ]

        for i, poly in enumerate(polys):
            A[i,s] = A[s,i] = poly.contains(Point(self.start)) or poly.touches(Point(self.start))
            A[i,g] = A[g,i] = poly.contains(Point(self.goal))  or poly.touches(Point(self.goal))
            for j in range(i+1, len(polys)):
                if poly.intersects(polys[j]):
                    A[i,j] = A[j,i] = 1

        self.adj = A

    def build_conjugate(self):
        # assume self.adj already set
        s, g = self.adj.shape[0]-2, self.adj.shape[0]-1
        idx_map, regs, pts = {}, [], []

        for i in range(self.adj.shape[0]):
            for j in range(i, self.adj.shape[1]):
                if self.adj[i,j]:
                    vidx = len(regs)
                    idx_map[(i,j)] = idx_map[(j,i)] = vidx

                    if i==s or j==s:
                        regs.append("start"); pts.append(self.start)
                    elif i==g or j==g:
                        regs.append("goal");  pts.append(self.goal)
                    else:
                        r1, r2 = self.regions[i], self.regions[j]
                        HS = np.vstack((r1.halfspaces, r2.halfspaces))
                        nrm= np.linalg.norm(HS[:,:-1], axis=1, keepdims=True)
                        A_ub = np.hstack((HS[:,:-1], nrm))
                        b_ub = -HS[:,-1]
                        c     = np.zeros(HS.shape[1]); c[-1] = -1
                        x0    = linprog(c, A_ub=A_ub, b_ub=b_ub,
                                        bounds=[(None,None)]*HS.shape[1],
                                        method="highs").x[:-1]
                        regs.append(HalfspaceIntersection(HS, x0))
                        pts.append(x0)

        self.gcs_regs   = regs
        self.region_pts = np.array(pts)

        m = len(regs)
        G = np.zeros((m,m), int)
        for (i,j), v in idx_map.items():
            for k in range(self.adj.shape[0]):
                if k not in (i,j):
                    if self.adj[i,k]:
                        G[v, idx_map[(i,k)]] = 1
                    if self.adj[j,k]:
                        G[v, idx_map[(j,k)]] = 1
        self.conjugate = G

    def solve(self, num_rounds=10):
        # 1) build adj / conjugate
        if not self._skip_overlap:
            self.build_overlap_adj()
        self.build_conjugate()

        # 2) solve the SOC‐relaxed GCS
        s, g = self.gcs_regs.index("start"), self.gcs_regs.index("goal")
        LMAX = np.linalg.norm(self.world_bounds[:,1] - self.world_bounds[:,0]) + 1.0

        # create variables
        y = {}; z = {}; phi = {}; l = {}
        for i in range(self.conjugate.shape[0]):
            for j in range(self.conjugate.shape[1]):
                if self.conjugate[i,j]:
                    y[i,j]   = cp.Variable(2)
                    z[i,j]   = cp.Variable(2)
                    phi[i,j] = (cp.Variable(integer=True)
                                if self.solve_integer else cp.Variable())
                    l[i,j]   = cp.Variable(nonneg=True)

        cons = []
        # SOC + perspective
        for (i,j), yij in y.items():
            zij    = z[(i,j)]
            phij   = phi[(i,j)]
            lij    = l[(i,j)]
            cons += [
                cp.SOC(lij, yij - zij),          # ||y−z|| ≤ l
                lij <= LMAX * phij               # l ≤ Lmax * φ
            ]
            # membership in region i and j
            for vec, v in ((yij, i), (zij, j)):
                if v == s:
                    cons.append(vec == phij * self.start)
                elif v == g:
                    cons.append(vec == phij * self.goal)
                else:
                    H = self.gcs_regs[v].halfspaces
                    A_hs, b_hs = H[:,:-1], -H[:,-1]
                    cons.append(A_hs @ vec <= b_hs * phij)

        # flow conservation + continuity
        for v in range(self.conjugate.shape[0]):
            ins  = [phi[(u,v)] for u in range(self.conjugate.shape[0]) if self.conjugate[u,v]]
            outs = [phi[(v,w)] for w in range(self.conjugate.shape[1]) if self.conjugate[v,w]]
            fin  = sum(ins) + (1 if v==s else 0)
            fout = sum(outs) + (1 if v==g else 0)
            cons += [fin == fout, fout <= 1]
            if v not in (s,g):
                cons.append(
                    sum(z[(u,v)] for u in range(self.conjugate.shape[0]) if self.conjugate[u,v])
                    == sum(y[(v,w)] for w in range(self.conjugate.shape[1]) if self.conjugate[v,w])
                )

        # bounds on φ
        for ph in phi.values():
            cons += [ph >= 0, ph <= 1]

        # solve relaxation
        obj = cp.sum([lval for lval in l.values()])
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=cp.MOSEK, verbose=True)

        # 3) recover node points X[v]
        X = np.zeros((self.conjugate.shape[0], 2))
        for v in range(self.conjugate.shape[0]):
            ins  = [(u,v) for u in range(self.conjugate.shape[0]) if self.conjugate[u,v]]
            outs = [(v,w) for w in range(self.conjugate.shape[1]) if self.conjugate[v,w]]
            if v == g:
                num = sum(z[e].value for e in ins)
                den = sum(phi[e].value for e in ins)
            else:
                num = sum(y[e].value for e in outs)
                den = sum(phi[e].value for e in outs)
            if den > 1e-8:
                X[v] = num / den

        # 4) randomized DFS rounding
        def path_length(path):
            pts = X[path]
            return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))

        best_path, best_cost = None, np.inf
        for _ in range(num_rounds):
            path = [s]
            seen = {s}
            while path[-1] != g:
                v = path[-1]
                nbrs = np.nonzero(self.conjugate[v])[0]
                weights = np.array([phi[(v,u)].value for u in nbrs])
                # filter
                valid = [(u,w) for u,w in zip(nbrs, weights) if w>1e-6 and u not in seen]
                if not valid:
                    path.pop()
                    if not path:
                        break
                    continue
                us, ws = zip(*valid)
                probs = np.array(ws) / np.sum(ws)
                next_v = random.choices(us, probs)[0]
                path.append(next_v)
                seen.add(next_v)

            if path and path[-1]==g:
                cost = path_length(path)
                if cost < best_cost:
                    best_cost, best_path = cost, list(path)

        if best_path is None:
            print("Rounding failed.")
            return []

        return [X[v] for v in best_path]




