import igl 
import numpy as np

class MeshDataset():
    def __init__(self):
        self.v,self.f = igl.read_triangle_mesh('bunny.obj')
        self.aabb_tree = igl.AABB()
        self.aabb_tree.init(self.v,self.f)


    def df(self, pts, return_grad=False):
        dfs, _, closest_pts = self.aabb_tree.squared_distance(self.v, 
                                                              self.f,
                                                              pts)
        dfs = np.sqrt(dfs)
        if return_grad:
            grad = closest_pts - pts
            grad = grad / np.linalg.norm(grad)
            return dfs, grad
        else:
            return dfs

mesh = MeshDataset()
func = mesh.df
igl.sparse_voxel_grid(np.array([[0.0,0.0,0.0]]).astype(np.float64), lambda x: func(x), 1/128.0)
