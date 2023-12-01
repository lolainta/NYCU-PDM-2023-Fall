import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def pairwise_registration(source, target, init, voxel_size):
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source,
        target,
        voxel_size * 0.8,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    icp_fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        voxel_size * 0.4,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return icp_fine.transformation


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


def o3d_voxel_down(pcd, voxel_size, verbose=False):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    if verbose:
        tqdm.write(
            "Downsample from %d points to %d points"
            % (len(pcd.points), len(pcd_down.points))
        )
    return pcd_down, pcd_fpfh


def custom_voxel_down(pcd, voxel_size, verbose=False):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    if verbose:
        tqdm.write(
            "Downsample from %d points to %d points"
            % (len(pcd.points), len(pcd_down.points))
        )
    return pcd_down, pcd_fpfh


def p2m(pos):
    pos[3] = -pos[3]
    ret = np.eye(4)
    ret[:3, 3] = pos[:3]
    ret[:3, :3] = R.from_quat(pos[3:]).as_matrix()
    return ret


def gen_refs():
    rots = [
        np.array(
            [
                [np.cos(np.deg2rad(deg)), 0, -np.sin(np.deg2rad(deg)), 0],
                [0, 1, 0, 0],
                [np.sin(np.deg2rad(deg)), 0, np.cos(np.deg2rad(deg)), 0],
                [0, 0, 0, 1],
            ]
        )
        for deg in range(10, 360, 10)
    ]
    mov = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -0.25],
            [0, 0, 0, 1],
        ]
    )
    dists = [rot @ [0, 0, -0.25, 0] for rot in rots]
    movs = [np.array(mov)]
    for dist in dists:
        tmp = np.eye(4)
        tmp[:, 3] = dist
        movs.append(tmp)
    refs = movs + [rots[0], rots[-1]]
    # for ref in refs:
    #     print(ref)
    return refs


def tune(mat):
    mat[0, 1] = 0
    mat[1, 0] = 0
    mat[1, 2] = 0
    mat[2, 1] = 0
    mat[1, 1] = 1
    mat[1, 3] = 0
    refs = gen_refs()
    for ref in refs:
        error = np.mean(np.linalg.norm(mat - ref))
        if error < 1e-2:
            mat = ref
            return mat
    else:
        tqdm.write(f"no {error}")
        tqdm.write(f"{mat}")
    return mat


def m2p(mat):
    mat = np.array(mat)

    ret = np.array([*mat[:3, 3], *R.from_matrix(mat[:3, :3]).as_quat()])
    # ret[0] = 10000 * ret[0]
    # ret[1] = 10000 * ret[1]
    # ret[2] = 10000 * ret[2]
    # # ret[1] = 0.12523484
    # ret[4] = 0
    # ret[5] = -ret[5]
    # ret[6] = 0
    # print(mat)
    # print(ret)
    # print(p2m(ret))
    # print(mat[:3, :3])
    ret = ret[[0, 1, 2, 6, 3, 4, 5]]
    # ret[3] = -ret[3]
    # print(ret)
    # print(p2m(ret))
    # print(mat)
    # print()
    return ret


def combine_pcd_from_pos(frames, cam_pos, color=[0, 1, 0]):
    traj = o3d.geometry.PointCloud()
    res = o3d.geometry.PointCloud() if frames else None
    for i in range(len(cam_pos)):
        ans = p2m(cam_pos[i])

        origin = o3d.geometry.PointCloud()
        origin.points.append([0, 0, 0])
        origin.transform(ans)
        traj += origin

        if frames:
            frames[i].pcd.transform(ans)
            res += frames[i].pcd
    traj.paint_uniform_color(color)
    return res, traj
