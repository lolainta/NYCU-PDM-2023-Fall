import numpy as np
import open3d as o3d
import argparse
import os
from multiprocessing import Semaphore, Process, Pool
from tqdm import trange, tqdm
from datetime import datetime
from Frame import Frame
from utils import draw_cumulative_res, predict_cam_pose, pairwise_registration
from my_icp import my_local_icp_algorithm
from queue import Queue
import copy
import random

sem = Semaphore(20)
results = Queue()
data_root = ""
frames = []


def global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    # draw_registration_result(source_down, target_down, result.transformation)
    return result.transformation


def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    # TODO: Use Open3D ICP function to implement
    raise NotImplementedError
    return result


def get_trans(s, t, args):
    sem.acquire()
    print(f"Gen transing for {s} {t}")
    src = Frame(s, args, voxel_size=1e-5, verbose=False)
    tar = Frame(t, args, voxel_size=1e-5, verbose=False)
    trans = global_registration(
        src.pcd_down,
        tar.pcd_down,
        src.pcd_fpfh,
        tar.pcd_fpfh,
        1e-4,
    )
    trans = my_local_icp_algorithm(src.pcd_down, tar.pcd_down, trans, 3e-5)
    print(f"{s}th trans done")
    sem.release()
    # results.put((s, trans))
    return s, trans


def reconstruct_without_pose_graph(frames, args):
    n_files = len(frames)
    transes = np.empty((n_files, 4, 4))
    for i in range(n_files - 1):
        get_trans(i, i + 1, args)
    # max_workers = max(1, min(multiprocessing.cpu_count() - 1, n_files))
    # mp_context = multiprocessing.get_context("spawn")
    # transes = np.load("transes.npy")
    # with multiprocessing.Pool(processes=max_workers) as pool:

    # plist = []
    # for i in range(n_files - 1):
    #     plist.append(Process(target=get_trans, args=(i, i + 1, args)))
    #     plist[-1].start()

    # for p in plist:
    #     p.join()
    # while not results.empty():
    #     i, trans = results.get()
    #     transes[i] = trans

    transes[-1] = np.identity(4)
    np.save("transes.npy", transes)
    print(transes)
    draw_cumulative_res(frames, transes)
    raise NotImplementedError
    return result_pcd, pred_cam_pos


def get_trans_ratio(
    s: int,
    t: int,
    voxel_size: float,
    method,
):
    src = frames[s]
    src_pcd = src.pcd_down
    src_fpfh = src.pcd_fpfh
    tar = frames[t]
    tar_pcd = tar.pcd_down
    tar_fpfh = tar.pcd_fpfh

    odo = global_registration(
        src_pcd,
        tar_pcd,
        src_fpfh,
        tar_fpfh,
        voxel_size * 15,
    )
    trans = method(
        src_pcd,
        tar_pcd,
        odo,
        voxel_size * 3,
    )
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        src_pcd,
        tar_pcd,
        voxel_size,
        trans,
    )
    ratio = info[5, 5] / (
        max(
            len(src_pcd.points),
            len(tar_pcd.points),
        )
    )
    return trans, info, ratio


def add_edge(g, s, t, voxel_size, method, node=False):
    trans, info, ratio = get_trans_ratio(s, t, voxel_size, method)
    tqdm.write(f"Register {s} {t} {ratio}")
    if s == t - 1:
        if node:
            g.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(trans))
            )
        g.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(s, t, trans, info, (s == t - 1))
        )
    if ratio > 0.3:
        g.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(s, t, trans, info, False)
        )
        return True
    return False


def gen_pose_graph(
    n: int,
    voxel_size: float,
    method=my_local_icp_algorithm,
):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    for src in trange(n):
        for tar in range(src + 1, n):
            if not add_edge(pose_graph, src, tar, voxel_size, method, node=True):
                break
    for tar in trange(n - 1, max(0, n - 10), -1):
        for src in range(tar - 1, max(tar - 10, -1), -1):
            if not add_edge(pose_graph, src, tar, voxel_size, method):
                break
    # for src in tqdm([random.randint(0, n - 1) for _ in range(10)]):
    #     for tar in [random.randint(0, n - 1) for _ in range(10)]:
    #         if not add_edge(pose_graph, src, tar, voxel_size, method):
    #             break
    return pose_graph


def reconstruct_with_pose_graph(n: int, voxel_size: float):
    # pose_graph = gen_pose_graph(n, voxel_size=voxel_size, method=my_local_icp_algorithm)
    pose_graph = gen_pose_graph(n, voxel_size=voxel_size, method=pairwise_registration)
    # pose_graph = o3d.io.read_pose_graph(os.path.join(args.data_root, "pose_graph.json"))
    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=voxel_size,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    # o3d.io.write_pose_graph(os.path.join(args.data_root, "pose_graph.json"), pose_graph)

    print("Transform points and display")
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
    pcds_down = copy.deepcopy([f.pcd_down for f in frames])
    pcd_combined = o3d.geometry.PointCloud()
    pred_cam_pos = []
    for point_id in range(n):
        origin_pcd = o3d.geometry.PointCloud()
        origin_pcd.points.append([0, 0, 0])
        origin_pcd.paint_uniform_color([1, 0, 0])
        pcds_down[point_id] += origin_pcd
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(to_real)
        pred_cam_pos.append(
            np.concatenate([pcds_down[point_id].points[-1], np.array([1, 0, 0, 0])]),
        )
        pcd_combined += pcds_down[point_id]
    o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined)
    # o3d.visualization.draw_geometries([pcd_combined])
    pred_cam_pos = np.asarray(pred_cam_pos)
    pred2 = predict_cam_pose(pose_graph)
    # print(pred_cam_pos.shape, pred2.shape)
    pred_cam_pos[:, :3] = pred2[:, :3]
    pred_cam_pos[:, 3:] = pred2[:, 3:]
    return pcd_combined, np.asarray(pred_cam_pos)


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
    n = 100
    voxel_size = 6e-6
    global frames
    frames = [
        Frame(i, data_root, voxel_size=voxel_size, verbose=True) for i in trange(n)
    ]
    return reconstruct_with_pose_graph(n, voxel_size)
    return reconstruct_without_pose_graph(frames, args)


if __name__ == "__main__":
    start_time = datetime.now()
    print(start_time)
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--floor", type=int, default=1)
    parser.add_argument(
        "-v", "--version", type=str, default="my_icp", help="open3d or my_icp"
    )
    parser.add_argument("--data_root", type=str, default="data_collection/first_floor/")
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

    # gt_cam_pos = np.load(os.path.join(args.data_root, "GT_pose.npy"))
    gt_cam_pos = np.load(os.path.join(args.data_root, "GT_pose.npy"))[:100]
    for i in range(len(gt_cam_pos)):
        print(gt_cam_pos[i, :3], gt_cam_pos[i, 3:])
        print(pred_cam_pos[i, :3], pred_cam_pos[i, 3:])
        print(np.linalg.norm(gt_cam_pos[i, :3] - pred_cam_pos[i, :3]))
        print()

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

    cam_traj_gt = o3d.geometry.PointCloud()
    for i in range(len(gt_cam_pos)):
        cam_traj_gt.points.append(gt_cam_pos[i, :3])
    cam_traj_gt.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([result_pcd, cam_traj_gt])
    end_time = datetime.now()
    print(f"Time: {end_time - start_time}")
