import open3d as o3d
import numpy as np
from copy import deepcopy
from tqdm import trange, tqdm
from utils import global_registration, predict_cam_pose
from Frame import Frame
import os


class Reconstructor:
    def __init__(
        self, id, sid, eid, data: list, method, voxel_size=5e-6, verbose=False
    ) -> None:
        self.id = id
        self.sid = sid
        self.eid = eid
        self.method = method
        self.n = len(data)
        if self.id != -1:
            assert self.n == self.eid - self.sid
        self.data = data
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.voxel_size = voxel_size
        self.verbose = verbose

    def reconstruct(self, read=True):
        ideneity = f"{self.id}_{self.sid}_{self.eid}_{self.method.__name__}"
        os.makedirs("results", exist_ok=True)
        pcd_path = os.path.join("results", f"{ideneity}.pcd")
        npy_path = os.path.join("results", f"{ideneity}.npy")
        if os.path.exists(pcd_path) and os.path.exists(npy_path) and read:
            tqdm.write(f"Reconstructor {ideneity} found")
            self.pcd = o3d.io.read_point_cloud(pcd_path)
            self.cam_pose = np.load(npy_path)
            return self.pcd, self.cam_pose
        tqdm.write(f"Reconstructor {ideneity} not found")
        if self.verbose:
            tqdm.write(f"Reconstructor {ideneity} start")
        self.pcd, self.cam_pose = self.reconstruct_with_pose_graph()
        if self.verbose:
            tqdm.write(f"Reconstructor {ideneity} end")
        o3d.io.write_point_cloud(os.path.join("results", f"{ideneity}.pcd"), self.pcd)
        np.save(os.path.join("results", f"{ideneity}.npy"), self.cam_pose)
        return self.pcd, self.cam_pose

    def reconstruct_with_pose_graph(self):
        # pose_graph = gen_pose_graph(n, voxel_size=voxel_size, method=my_local_icp_algorithm)
        pose_graph_path = os.path.join(
            "results", f"{self.id}_{self.sid}_{self.eid}.json"
        )
        if os.path.exists(pose_graph_path):
            self.pose_graph = o3d.io.read_pose_graph(pose_graph_path)
        else:
            self.gen_pose_graph()

        # pose_graph = o3d.io.read_pose_graph(os.path.join(args.data_root, "pose_graph.json"))
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.voxel_size,
            edge_prune_threshold=0.25,
            reference_node=0,
        )
        o3d.pipelines.registration.global_optimization(
            self.pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

        # o3d.io.write_pose_graph(os.path.join(args.data_root, "pose_graph.json"), pose_graph)
        # pcds = copy.deepcopy([f.pcd for f in frames])
        # for point_id in range(len(pcds)):
        #     pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        # o3d.visualization.draw_geometries(pcds)

        to_real = [
            [10000, 0, 0, 0],
            [0, 10000, 0, 0.125],
            [0, 0, -10000, -0.25],
            [0, 0, 0, 1],
        ]
        to_real2 = [
            [0, 0, 1, -0.25 / 10000],
            [0, 1, 0, -0.125 / 10000],
            [-1, 0, 0, 1.25 / 10000],
            [0, 0, 0, 1 / 10000],
        ]
        pcds_down = deepcopy([f.pcd_down for f in self.data])
        pcd_combined = o3d.geometry.PointCloud()
        pred_cam_pos = []
        for point_id in range(self.n):
            origin_pcd = o3d.geometry.PointCloud()
            origin_pcd.points.append([0, 0, 0])
            origin_pcd.paint_uniform_color([1, 0, 0])
            pcds_down[point_id] += origin_pcd

            pcds_down[point_id].transform(self.pose_graph.nodes[point_id].pose)
            pcds_down[point_id].transform(to_real2)

            pred_cam_pos.append(
                np.concatenate(
                    [pcds_down[point_id].points[-1], np.array([1, 0, 0, 0])]
                ),
            )
            pcd_combined += pcds_down[point_id]
        pred_cam_pos = np.asarray(pred_cam_pos)
        # o3d.visualization.draw_geometries([pcd_combined])
        pred2 = predict_cam_pose(self.pose_graph)
        # print(pred_cam_pos.shape, pred2.shape)
        pred_cam_pos[:, :3] = pred2[:, :3]
        pred_cam_pos[:, 3:] = pred2[:, 3:]
        return pcd_combined, pred_cam_pos

    def get_trans_ratio(self, s: int, t: int):
        src = self.data[s]
        src_pcd = src.pcd_down
        src_fpfh = src.pcd_fpfh
        tar = self.data[t]
        tar_pcd = tar.pcd_down
        tar_fpfh = tar.pcd_fpfh

        trans = global_registration(
            src_pcd,
            tar_pcd,
            src_fpfh,
            tar_fpfh,
            self.voxel_size * 15,
        )
        trans = self.method(
            src_pcd,
            tar_pcd,
            trans,
            self.voxel_size * 3,
        )
        for _ in range(4):
            trans = self.method(
                src_pcd,
                tar_pcd,
                trans,
                self.voxel_size,
            )
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            src_pcd,
            tar_pcd,
            self.voxel_size,
            trans,
        )
        ratio = info[5, 5] / (
            max(
                len(src_pcd.points),
                len(tar_pcd.points),
            )
        )
        return trans, info, ratio

    def add_edge(self, s, t, node=False):
        trans, info, ratio = self.get_trans_ratio(s, t)
        if self.verbose:
            tqdm.write(f"Register {self.id}: {s}-{t} {ratio}")
        if s == t - 1:
            if node:
                self.pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(trans))
                )
            self.pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    s, t, trans, info, (s != t - 1)
                )
            )
        if ratio > 0.6:
            self.pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(s, t, trans, info, True)
            )
        return ratio > 0.4

    def gen_pose_graph(self):
        odometry = np.identity(4)
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
        for src in trange(self.n):
            for tar in range(src + 1, min(src + 5, self.n)):
                if not self.add_edge(src, tar, node=True):
                    break
                    pass
        # for tar in trange(self.n - 1, max(0, self.n - 4), -1):
        #     for src in range(tar - 1, max(tar - 4, -1), -1):
        #         if not self.add_edge(src, tar):
        #             break
        #             pass
        # for src in tqdm([random.randint(0, n - 1) for _ in range(10)]):
        #     for tar in [random.randint(0, n - 1) for _ in range(10)]:
        #         add_edge(pose_graph, src, tar, voxel_size, method)
