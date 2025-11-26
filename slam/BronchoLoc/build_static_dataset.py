import pyvista as pv
import trimesh
import numpy as np
import networkx as nx
import scipy.ndimage
import os
import torch 

class LungGraphBuilder:
    def __init__(self, vtk_path, cad_path):
        self.vtk_path = vtk_path
        self.cad_path = cad_path
        self.raw_graph = nx.Graph()
        self.final_graph = nx.Graph()
        self.sdf_grid = None
        self.voxel_transform = None
        self.voxel_pitch = None

    def process(self):
        print(f"[1/5] Loading VTK skeleton: {self.vtk_path}")
        skeleton = pv.read(self.vtk_path)
        
        print(f"[2/5] Loading CAD and computing SDF: {self.cad_path}")
        self._compute_sdf_from_cad()
        
        print("[3/5] Parsing VTK connectivity...")
        self._vtk_to_raw_graph(skeleton)
        
        print(f"[4/5] Simplifying topology (Raw nodes: {len(self.raw_graph.nodes)})...")
        self._simplify_graph_topology(skeleton.points)
        
        print("[5/5] Injecting radius data from CAD SDF...")
        self._inject_radius_data()
        
        return self.final_graph

    def _compute_sdf_from_cad(self):
        # Load mesh
        mesh = trimesh.load(self.cad_path)
        
        # --- AUTO-SCALE FIX ---
        extents = mesh.extents
        max_dim = np.max(extents)
        
        # Target ~256 voxels for precision
        target_resolution = 256
        self.voxel_pitch = max_dim / target_resolution
        
        print(f"      Detected Max Dimension: {max_dim:.4f}")
        print(f"      Calculated Dynamic Pitch: {self.voxel_pitch:.6f} units")
        
        # Voxelize
        voxel_grid = mesh.voxelized(pitch=self.voxel_pitch).fill()
        self.voxel_transform = voxel_grid.transform
        matrix = voxel_grid.matrix
        
        print(f"      SDF Grid generated. New Shape: {matrix.shape}")
        
        # Compute Distance Transform
        self.sdf_grid = scipy.ndimage.distance_transform_edt(matrix) * self.voxel_pitch

    def _get_radius_at_point(self, point_3d):
        mat = np.linalg.inv(self.voxel_transform)
        val = np.dot(mat, np.append(point_3d, 1.0))
        idx = np.round(val[:3]).astype(int)
        
        if (0 <= idx[0] < self.sdf_grid.shape[0] and 
            0 <= idx[1] < self.sdf_grid.shape[1] and 
            0 <= idx[2] < self.sdf_grid.shape[2]):
            return self.sdf_grid[idx[0], idx[1], idx[2]]
        return 0.0

    def _vtk_to_raw_graph(self, skeleton):
        lines = skeleton.lines
        i = 0
        while i < len(lines):
            n_points = lines[i]
            segment_indices = lines[i+1 : i+1+n_points]
            for k in range(len(segment_indices)-1):
                u, v = segment_indices[k], segment_indices[k+1]
                self.raw_graph.add_edge(u, v)
            i += (n_points + 1)

    def _simplify_graph_topology(self, all_points):
        junctions = [n for n in self.raw_graph.nodes if self.raw_graph.degree(n) != 2]
        for n in junctions:
            self.final_graph.add_node(n, pos=all_points[n])
            
        visited_edges = set()
        for start_node in junctions:
            for neighbor in self.raw_graph.neighbors(start_node):
                edge_id = tuple(sorted((start_node, neighbor)))
                if edge_id in visited_edges: continue
                
                path = [start_node, neighbor]
                curr = neighbor
                prev = start_node
                
                while self.raw_graph.degree(curr) == 2:
                    nbrs = list(self.raw_graph.neighbors(curr))
                    next_node = nbrs[0] if nbrs[0] != prev else nbrs[1]
                    prev = curr
                    curr = next_node
                    path.append(curr)
                
                end_node = curr
                for k in range(len(path)-1):
                    visited_edges.add(tuple(sorted((path[k], path[k+1]))))
                
                geometry = all_points[path]
                self.final_graph.add_edge(start_node, end_node, pts=geometry)
        print(f"      Final Graph: {len(self.final_graph.nodes)} Bifurcations, {len(self.final_graph.edges)} Airways.")

    def _inject_radius_data(self):
        for u, v, data in self.final_graph.edges(data=True):
            points = data['pts']
            radii = [self._get_radius_at_point(p) for p in points]
            data['radius'] = np.array(radii)

    def export_static_dataset(self, output_dir="static"):
        """
        Exports all three files needed for the Deep-Lung-ST Network:
        1. deep_lung_graph.npz (Graph Topology & Features)
        2. lung_sdf.pt (Differentiable Volume)
        3. grid_transform.npy (World-to-Voxel Matrix)
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"Exporting static dataset to '{output_dir}/'...")

        # --- 1. Export Graph ---
        node_list = list(self.final_graph.nodes)
        node_map = {n: i for i, n in enumerate(node_list)}
        edge_index = []
        edge_attr = []
        
        for u, v, data in self.final_graph.edges(data=True):
            idx_u = node_map[u]
            idx_v = node_map[v]
            edge_index.append([idx_u, idx_v])
            edge_index.append([idx_v, idx_u])
            
            pts = data['pts']
            radii = data['radius']
            length = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
            avg_rad = np.mean(radii)
            # 13-dim feature: Start(3), End(3), Mid(3), Rad(1), Len(1)
            feat = np.concatenate([pts[0], pts[-1], pts[len(pts)//2], [avg_rad], [length]])
            edge_attr.append(feat)
            edge_attr.append(feat)

        npz_path = os.path.join(output_dir, "deep_lung_graph.npz")
        np.savez(npz_path, 
                 edge_index=np.array(edge_index).T,
                 edge_attr=np.array(edge_attr),
                 node_pos=np.array([self.final_graph.nodes[n]['pos'] for n in node_list]))
        print(f"  - Graph saved to {npz_path}")

        # --- 2. Export SDF Volume (Torch Tensor) ---
        sdf_path = os.path.join(output_dir, "lung_sdf.pt")
        # Convert to float32 Tensor for PyTorch grid_sample
        sdf_tensor = torch.from_numpy(self.sdf_grid).float()
        torch.save(sdf_tensor, sdf_path)
        print(f"  - SDF Volume saved to {sdf_path}")

        # --- 3. Export Grid Transform ---
        trans_path = os.path.join(output_dir, "grid_transform.npy")
        np.save(trans_path, self.voxel_transform)
        print(f"  - Grid Transform saved to {trans_path}")

    def visualize(self, save_screenshot="verification.png"):
        print("Preparing visualization...")
        p = pv.Plotter(off_screen=False) 

        # 1. Plot Original CAD 
        if os.path.exists(self.cad_path):
            mesh = pv.read(self.cad_path)
            p.add_mesh(mesh, color='wheat', opacity=0.25, label='CAD Wall')
        
        # 2. Plot Nodes
        node_pts = np.array([self.final_graph.nodes[n]['pos'] for n in self.final_graph.nodes])
        if len(node_pts) > 0:
            p.add_mesh(pv.PolyData(node_pts), color='red', point_size=12, 
                       render_points_as_spheres=True, label='Graph Nodes')
        
        # 3. Plot Edges (SDF Tubes)
        tubes = []
        for u, v, data in self.final_graph.edges(data=True):
            pts = data['pts']
            line = pv.lines_from_points(pts)
            rad = np.mean(data['radius'])
            if rad < 0.1: rad = 0.5 
            
            tube = line.tube(radius=rad)
            tubes.append(tube)
            
        if tubes:
            combined_tubes = tubes[0].merge(tubes[1:])
            p.add_mesh(combined_tubes, color='blue', opacity=0.4, label='Graph Edges (SDF)')

        # 4. Plot Centerline
        if os.path.exists(self.vtk_path):
            skel = pv.read(self.vtk_path)
            p.add_mesh(skel, color='lime', line_width=4, render_lines_as_tubes=True, label='Centerline')

        p.add_legend()
        p.add_axes()
        
        try:
            p.show()
        except Exception:
            print(f"Display not available. Saving screenshot to {save_screenshot}")
            p.show(screenshot=save_screenshot)

if __name__ == "__main__":
    # Ensure these paths match your files
    vtk_file = "patient/centerline.vtk" 
    cad_file = "patient/lungs.obj" 
    
    builder = LungGraphBuilder(vtk_file, cad_file)
    G = builder.process()
    
    # Creates the 'static' folder with all 3 required files
    builder.export_static_dataset(output_dir="static")
    
    # Optional check
    builder.visualize()