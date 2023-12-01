import numpy as np
from tqdm import tqdm, trange
import copy
import open3d as o3d
from utils import draw_registration_result
from sklearn.neighbors import KDTree


def find_nearest_neighbors(source_pc, target_pc, nearest_neigh_num):
    # Find the closest neighbor for each anchor point through KDTree
    point_cloud_tree = o3d.geometry.KDTreeFlann(source_pc)
    # Find nearest target_point neighbor index
    points_arr = []
    for point in target_pc.points:
        [_, idx, _] = point_cloud_tree.search_knn_vector_3d(point, nearest_neigh_num)
        points_arr.append(source_pc.points[idx[0]])
    return np.asarray(points_arr)


def my_local_icp_algorithm(src, tar, transform_matrix, voxel_size):
    source = copy.deepcopy(src)
    target = copy.deepcopy(tar)
    source = src
    target = tar
    target_points = np.asarray(target.points)
    source = source.transform(transform_matrix)

    curr_iteration = 0
    curr_cost = 1
    prev_cost = 10

    while True:
        # 1. Find nearest neighbors
        new_source_points = find_nearest_neighbors(source, target, 100)

        # 2. Find point cloud centroids and their repositions
        source_centroid = np.mean(new_source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        source_repos = np.zeros_like(new_source_points)
        target_repos = np.zeros_like(target_points)
        source_repos = np.asarray(
            [
                new_source_points[ind] - source_centroid
                for ind in range(len(new_source_points))
            ]
        )
        target_repos = np.asarray(
            [target_points[ind] - target_centroid for ind in range(len(target_points))]
        )

        # 3. Find correspondence between source and target point clouds
        cov_mat = target_repos.transpose() @ source_repos

        U, X, Vt = np.linalg.svd(cov_mat)
        R = U @ Vt
        t = target_centroid - R @ source_centroid
        t = np.reshape(t, (1, 3))
        curr_cost = np.linalg.norm(target_repos - (R @ source_repos.T).T)
        tqdm.write(f"Curr_cost={curr_cost}")
        if (prev_cost - curr_cost) > voxel_size:
            prev_cost = curr_cost
            transform_matrix = np.hstack((R, t.T))
            transform_matrix = np.vstack((transform_matrix, np.array([0, 0, 0, 1])))
            # If cost_change is acceptable, update source with new transformation matrix
            source = source.transform(transform_matrix)
            curr_iteration += 1
        else:
            break
    tqdm.write(f"Iteration={curr_iteration}")
    # Visualize final iteration and print out final variables
    # draw_registration_result(source, target, transform_matrix)
    return transform_matrix


# def my_local_icp_algorithm(source_down, target_down, init_trans, tolerance=1e-5):
#     source_down = np.asarray(source_down.points)
#     target_down = np.asarray(target_down.points)
#     # tqdm.write(f"source: {len(source_down)}, target: {len(target_down)}")
#     # for opt in opts:
#     #     error = compute_error(source_down, target_down, opt)
#     #     if error < tolerance:
#     #         tqdm.write(f"Converged from opt {opt}, {error}")
#     #         del source_down, target_down
#     #         return opt

#     transformation = init_trans
#     init_error = compute_error(source_down, target_down, transformation)
#     tqdm.write(f"Init error: {init_error}")
#     error = 1
#     for _ in range(5):
#         closest_points = closest_point_matching(source_down, target_down)
#         transformation_update = compute_transform(
#             source_down, target_down, closest_points
#         )
#         transformation = np.dot(transformation_update, transformation)

#         source_transformed = (
#             np.dot(source_down, transformation[:3, :3].T) + transformation[:3, 3]
#         )
#         error = np.mean(
#             np.linalg.norm(source_transformed - target_down[closest_points], axis=1)
#         )
#         if error < tolerance:
#             tqdm.write(f"Converged {error}")
#             break
#     else:
#         tqdm.write(f"Did not converge {error}")
#         if init_error < error:
#             return init_trans
#     return transformation
