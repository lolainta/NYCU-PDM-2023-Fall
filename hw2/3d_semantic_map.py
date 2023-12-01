import numpy as np
import open3d as o3d
import argparse
from datetime import datetime
import os
from reconstruct import Frame
from tqdm import trange, tqdm
from reconstruct.utils import (
    o3d_voxel_down,
    custom_voxel_down,
    combine_pcd_from_pos,
    m2p,
    tune,
    gen_refs,
)
from reconstruct.Frame import BaseFrameType
from copy import deepcopy

np.set_printoptions(threshold=200, precision=3, suppress=True)


def reconstruct(args, init_pose):
    data_root = args.data_root
    n = len(os.listdir(os.path.join(data_root, "rgb")))
    assert n == len(os.listdir(os.path.join(data_root, "depth")))
    voxel_size = 3e-2
    voxel_down = custom_voxel_down if args.voxel_down == "custom" else o3d_voxel_down

    print(f"n: {n}, voxel_size: {voxel_size}, voxel_down: {args.voxel_down}")
    frames = [
        Frame(
            id=i,
            data_root=data_root,
            img_type=BaseFrameType.SEMENTIC if args.gt else BaseFrameType.SEMENTIC_PRED,
            voxel_down=voxel_down,
            voxel_size=voxel_size,
            verbose=False,
        )
        for i in trange(0, 0 + n)
    ]
    print(
        f"Frames loaded from {os.path.realpath(frames[0].img_path)} to {os.path.realpath(frames[-1].img_path)}"
    )

    acc_trans = [np.eye(4)]
    for i in trange(n - 1):
        src = frames[i]
        tar = frames[i + 1]
        # trans = global_registration(
        #     src.pcd_down,
        #     tar.pcd_down,
        #     src.pcd_fpfh,
        #     tar.pcd_fpfh,
        #     voxel_size=voxel_size * 8,
        # )
        # for ratio in [8, 5, 4, 3, 2, 1, 1, 1, 0.8]:
        #     trans = o3d.pipelines.registration.registration_icp(
        #         src.pcd_down,
        #         tar.pcd_down,
        #         voxel_size * ratio,
        #         trans,
        #         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        #     ).transformation
        # info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        #     src.pcd_down,
        #     tar.pcd_down,
        #     voxel_size,
        #     trans,
        # )
        # ratio = info[5, 5] / (
        #     max(
        #         len(src.pcd_down.points),
        #         len(tar.pcd_down.points),
        #     )
        # )
        cand = []
        trans = np.eye(4)
        for ref in gen_refs():
            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                src.pcd_down,
                tar.pcd_down,
                voxel_size,
                ref,
            )
            ratio = info[5, 5] / (
                max(
                    len(src.pcd_down.points),
                    len(tar.pcd_down.points),
                )
            )
            cand.append((ref, ratio))
        # tqdm.write([c[1] for c in cand])
        cid = np.argmax([c[1] for c in cand])
        trans = cand[cid][0]
        ratio = cand[cid][1]
        trans = deepcopy(trans)
        trans = tune(trans)
        acc_trans.append(trans)
        # tqdm.write(f"Register {i}-{i+1} {ratio}")
        # tqdm.write(str(trans))
    cam_pose = []
    cur = np.eye(4)
    for t in acc_trans:
        cur = np.matmul(cur, t)
        cam_pose.append(m2p(cur))
    cam_pose = np.array(cam_pose)
    cam_pose[:, :3] += init_pose[:3]

    # cam_pose = np.load(os.path.join(data_root, "GT_pose.npy"))
    # cam_pose = cam_pose[:n]
    pcd, traj = combine_pcd_from_pos(frames, cam_pose, [0, 0, 1])
    return pcd, traj, cam_pose


if __name__ == "__main__":
    start_time = datetime.now()
    print(start_time)
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--floor", type=int, default=1, choices=[1, 2])
    parser.add_argument(
        "-v", "--version", type=str, default="my_icp", help="open3d or my_icp"
    )
    parser.add_argument(
        "--voxel_down", type=str, default="custom", choices=["custom", "open3d"]
    )
    parser.add_argument(
        "--gt", action="store_true", help="use ground truth semantic images"
    )
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"
    data_root = args.data_root
    if args.floor == 1:
        init_pos = [0, 0.125, -0.25, -1, 0, 0, 0]
    elif args.floor == 2:
        init_pos = [-0.233, 2.925, -1.221, -1, 0, 0, 0]

    result_pcd, result_traj, result_pos = reconstruct(args, init_pos)
    end_time = datetime.now()
    print("Time: ", end_time - start_time)

    gt_cam_pose = np.load(os.path.join(data_root, "GT_pose.npy"))
    gt_cam_pose[:, 3] *= -1
    gt_cam_pose = gt_cam_pose[0 : 0 + len(result_pos)]
    # print(gt_cam_pose.shape, result_pos.shape)
    # for i in range(len(gt_cam_pose)):
    #     print(gt_cam_pose[i], result_pos[i])
    m2d = np.mean(
        np.linalg.norm(gt_cam_pose[:, :3] - result_pos[:, :3])
        + np.linalg.norm(np.abs(gt_cam_pose[:, 3:]) - np.abs(result_pos[:, 3:]))
    )
    print("Mean L2 distance: ", m2d)
    _, gt_traj = combine_pcd_from_pos(None, gt_cam_pose, [1, 0, 0])

    if args.floor == 1:
        lb, ub = -0.1, 0.9
    elif args.floor == 2:
        lb, ub = 0.8, 3.5
    box = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(
            np.array(
                [
                    [100, lb, 100],
                    [100, lb, -100],
                    [-100, lb, -100],
                    [-100, lb, 100],
                    [100, ub, 100],
                    [100, ub, -100],
                    [-100, ub, -100],
                    [-100, ub, 100],
                ]
            )
        )
    )
    result_pcd = result_pcd.crop(box)

    o3d.visualization.draw_geometries([result_pcd, result_traj, gt_traj])
