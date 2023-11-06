import copy
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def draw_cumulative_res(frames, trans):
    # frames = [frames[0], frames[-1]]
    # trans = [trans[0], trans[-1]]
    assert len(frames) == len(trans), f"{len(frames)} {len(trans)}"
    # for fr, tr in zip(frames, trans):
    #     fr.pcd.transform(tr)
    res = frames[0].pcd
    for i in range(1, len(frames)):
        res.transform(trans[i - 1])
        res += frames[i].pcd
    o3d.visualization.draw_geometries([res])


def predict_cam_pose(graph):
    n = len(graph.nodes)

    def p2m(pos):
        ret = np.eye(4)
        ret[:3, 3] = pos[:3]
        ret[:3, :3] = R.from_quat(pos[3:]).as_matrix()
        return ret

    def m2p(mat):
        ret = np.array([*mat[:3, 3], *R.from_matrix(mat[:3, :3]).as_quat()])
        ret[0] = 10000 * ret[0]
        ret[1] = 10000 * ret[1]
        ret[2] = 10000 * ret[2]
        # ret[1] = 0.12523484
        ret[4] = 0
        ret[5] = -ret[5]
        ret[6] = 0
        return ret

    init_pos = p2m(np.array([0, 0.000012, -0.000025, 1, 0, 0, 0]))
    # print(init_pos)
    # print(o3d.geometry.get_rotation_matrix_from_quaternion(np.array([1, 0, -0, 0])))
    pred_cam_pos = np.zeros((n, 7))
    # pred_cam_pos[0, :] = m2p(init_pos)
    # print(graph.nodes[0].pose)
    # print(np.linalg.inv(graph.nodes[0].pose))
    # print(graph.nodes[1].pose)
    # print(np.linalg.inv(graph.nodes[1].pose))
    # print(graph.nodes[2].pose)
    # print(np.linalg.inv(graph.nodes[2].pose))
    for i in range(0, n):
        cur = np.dot(
            np.linalg.inv(init_pos),
            graph.nodes[i].pose,
        )
        # cur = np.dot(init_pos, graph.nodes[i].pose)
        # cur = np.dot(init_pos, np.linalg.inv(graph.nodes[i].pose))
        # cur = copy.deepcopy(graph.nodes[i].pose)
        # print(cur)
        assert cur.shape == (4, 4)
        pred_cam_pos[i, :] = m2p(cur)
    return pred_cam_pos


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


def preprocess_point_cloud(pcd, voxel_size, verbose=False):
    # TODO: Do voxelization to reduce the number of points for less memory usage and speedup
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    # size = len(pcd.points)
    # if size > 1e6:
    #     voxel_size *= 10
    # while True:
    #     pcd_down = pcd.voxel_down_sample(voxel_size)
    #     size = len(pcd_down.points)
    #     tqdm.write(
    #         "Downsample from %d points to %d points"
    #         % (len(pcd.points), len(pcd_down.points))
    #     )
    #     tqdm.write(f"Voxel size: {voxel_size}")
    #     if size < 1e5:
    #         break
    #     voxel_size *= 2

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
