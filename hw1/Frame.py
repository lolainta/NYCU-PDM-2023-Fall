import open3d as o3d
import numpy as np
from tqdm import tqdm
from multiprocessing import Lock


def get_intrinsic():
    # return o3d.camera.PinholeCameraIntrinsic(
    #     o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    # )
    width, height = 512, 512
    fov = 90
    fx = width / (2 * np.tan(np.deg2rad(fov / 2)))
    fy = height / (2 * np.tan(np.deg2rad(fov / 2)))
    cx = width / 2
    cy = height / 2
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


locks = [Lock() for _ in range(1000)]


class Frame:
    def __init__(self, id: int, data_root: str, voxel_size=1e-5, verbose=False):
        self.id = id
        self.verbose = verbose
        self.rgb_path = f"{data_root}/rgb/{id+1}.png"
        self.depth_path = f"{data_root}/depth/{id+1}.png"
        self.rgbd_img, self.pcd = self.depth_image_to_point_cloud()
        self.pcd_down, self.pcd_fpfh = self.preprocess_point_cloud(voxel_size)

    def depth_image_to_point_cloud(self):
        # TODO: Get point cloud from rgb and depth image
        # locks[self.id].acquire()
        color_raw = o3d.io.read_image(self.rgb_path)
        depth_raw = o3d.io.read_image(self.depth_path)
        # locks[self.id].release()
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw,
            depth_raw,
            convert_rgb_to_intensity=False,
            # depth_scale=0.1,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, get_intrinsic()
        )
        return rgbd_image, pcd

    def preprocess_point_cloud(self, voxel_size):
        # TODO: Do voxelization to reduce the number of points for less memory usage and speedup
        # print(":: Downsample with a voxel size %.3f." % voxel_size)
        # size = len(self.pcd.points)
        # while size > 5000:
        #     voxel_size *= 1.1
        #     pcd_down = self.pcd.voxel_down_sample(voxel_size)
        #     size = len(pcd_down.points)
        #     tqdm.write(f":: Downsample {size} points")

        pcd_down = self.pcd.voxel_down_sample(voxel_size)
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
        if self.verbose:
            tqdm.write(
                ":: Downsample from %d points to %d points"
                % (len(self.pcd.points), len(pcd_down.points))
            )
        return pcd_down, pcd_fpfh
