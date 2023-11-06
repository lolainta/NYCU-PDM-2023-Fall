import numpy as np
import open3d as o3d
import argparse
import os
from tqdm import trange
from datetime import datetime
from Frame import Frame
from utils import pairwise_registration
from my_icp import my_local_icp_algorithm
from Reconstructor import Reconstructor
from Fragment import Fragment
import traceback
from multiprocessing.pool import ThreadPool as Pool

data_root = ""
start_time = datetime.now()


def reconstruct_fragment(sid, eid, data_root, voxel_size, method):
    r = Reconstructor(sid, eid, data_root, voxel_size, method, False)
    return r.reconstruct()


def fetch_frame(id, data_root, voxel_size, verbose):
    return Frame(id, data_root, voxel_size, verbose)


def worker(r: Reconstructor, read=True) -> Fragment:
    s = r.reconstruct(read)
    ret = Fragment(r.id, r.sid, r.eid)
    ret.set_data(s, 6e-2, False)
    o3d.visualization.draw_geometries([ret.pcd])
    return ret


def reconstruct_optimized(n):
    voxel_size = 4e-6
    k = 18
    print(f"n: {n}, voxel_size: {voxel_size}, k: {k}")
    frames = [Frame(i, data_root, voxel_size, False) for i in trange(n)]
    # frames = [None] * n
    rs = [
        Reconstructor(
            i // k, i, i + k, frames[i : i + k], pairwise_registration, voxel_size
        )
        for i in range(0, n - k, k)
    ]
    with Pool(8) as p:
        frags = p.map(worker, rs)
    print(len(frags))
    end_time = datetime.now()
    print(end_time)
    print(f"Time: {end_time - start_time}")

    r = Reconstructor(-1, 0, n, frags, pairwise_registration, 1e-2, verbose=False)
    total = worker(r)
    print(total.cam_pose)
    o3d.visualization.draw_geometries([total.pcd])
    return total.pcd, total.cam_pose


def reconstruct_normal(n):
    voxel_size = 6e-6
    print(f"n: {n}, voxel_size: {voxel_size}")
    # frames = [None] * n
    frames = [Frame(i, data_root, voxel_size, False) for i in trange(n)]
    r = Reconstructor(
        48763, 0, n, frames, pairwise_registration, voxel_size, verbose=False
    )
    total = worker(r, read=False)
    # o3d.visualization.draw_geometries([total.pcd])
    return total.pcd, total.cam_pose


def reconstruct(args):
    # TODO: Return results
    """
    For example:
        ...
        args.version == 'open3d':
            trans = local_icp_algorithm()
        args.version == 'my_icp':
            trans = my_local_icp_algorithm()
        ...
    """
    n = len(os.listdir(os.path.join(data_root, "rgb")))
    assert n == len(os.listdir(os.path.join(data_root, "depth")))
    return reconstruct_normal(n)
    return reconstruct_optimized(n)
    combined = o3d.geometry.PointCloud()
    for pcd in frags:
        o3d.visualization.draw_geometries([pcd])
        combined += pcd
    # o3d.visualization.draw_geometries([combined])
    return
    frags = [Fragment(i, k, frames[i : i + k]) for i in range(0, n, k)]
    return reconstruct_with_pose_graph(n, voxel_size)
    return reconstruct_without_pose_graph(frames, args)


if __name__ == "__main__":
    start_time = datetime.now()
    try:
        print(start_time)
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--floor", type=int, default=1)
        parser.add_argument(
            "-v", "--version", type=str, default="my_icp", help="open3d or my_icp"
        )
        parser.add_argument(
            "--data_root", type=str, default="data_collection/first_floor/"
        )
        args = parser.parse_args()

        if args.floor == 1:
            args.data_root = "data_collection/first_floor/"
        elif args.floor == 2:
            args.data_root = "data_collection/second_floor/"

        data_root = args.data_root
        # TODO: Output result point cloud and estimated camera pose
        """
        Hint: Follow the steps on the spec
        """
        result_pcd, pred_cam_pos = reconstruct(args)

        gt_cam_pos = np.load(os.path.join(args.data_root, "GT_pose.npy"))
        gt_cam_pos = gt_cam_pos[: len(pred_cam_pos)]
        # for i in range(len(gt_cam_pos)):
        #     print(gt_cam_pos[i, :3], gt_cam_pos[i, 3:])
        #     print(pred_cam_pos[i, :3], pred_cam_pos[i, 3:])
        #     print(np.linalg.norm(gt_cam_pos[i, :3] - pred_cam_pos[i, :3]))
        #     print()

        # TODO: Calculate and print L2 distance
        """
        Hint: Mean L2 distance = mean(norm(ground truth - estimated camera trajectory))
        """

        m2d = np.mean(
            np.linalg.norm(np.absolute(gt_cam_pos) - np.absolute(pred_cam_pos), axis=1)
        )
        print(f"Mean L2 distance: {m2d}")

        # TODO: Visualize result
        """
        Hint: Sould visualize
        1. Reconstructed point cloud
        2. Red line: estimated camera pose
        3. Black line: ground truth camera pose
        """
        # result_pcd.transform(
        #     np.linalg.inv(
        #         [
        #             [-1, 0, 0, 3 / 10000],
        #             [0, -1, 0, -0.1 / 10000],
        #             [0, 0, 1, -2 / 10000],
        #             [0, 0, 0, 1 / 10000],
        #         ]
        #     )
        # )
        # result_pcd.transform(
        #     [
        #         [-1, 0, 0, 0],
        #         [0, 1, 0, 0],
        #         [0, 0, -1, -2],
        #         [0, 0, 0, 1],
        #     ]
        # )

        cam_traj_gt = o3d.geometry.PointCloud()
        for i in range(len(gt_cam_pos)):
            cam_traj_gt.points.append(gt_cam_pos[i, :3])
        cam_traj_gt.paint_uniform_color([0, 0, 0])
    except Exception as e:
        traceback.print_exc()
        print(e)
    end_time = datetime.now()
    print(end_time)
    print(f"Time: {end_time - start_time}")

    xcord = o3d.geometry.PointCloud()
    for i in range(0, 5, 1):
        xcord.points.append([i, 0, 0])
    ycord = o3d.geometry.PointCloud()
    for i in range(0, 5, 1):
        ycord.points.append([0, i, 0])
    zcord = o3d.geometry.PointCloud()
    for i in range(0, 5, 1):
        zcord.points.append([0, 0, i])
    xcord.paint_uniform_color([1, 0, 0])
    ycord.paint_uniform_color([0, 1, 0])
    zcord.paint_uniform_color([0, 0, 1])
    cords = o3d.geometry.PointCloud()
    cords += xcord
    cords += ycord
    cords += zcord
    box = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(
            np.array(
                [
                    [100, -0.6, 100],
                    [100, -0.6, -100],
                    [-100, -0.6, -100],
                    [-100, -0.6, 100],
                    [100, 3, 100],
                    [100, 3, -100],
                    [-100, 3, -100],
                    [-100, 3, 100],
                ]
            )
        )
    )
    result_pcd = result_pcd.crop(box)

    o3d.visualization.draw_geometries([result_pcd, cam_traj_gt])
