import open3d as o3d
import numpy as np
from tqdm import tqdm
from multiprocessing import Lock
from utils import preprocess_point_cloud


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


intrinsic = get_intrinsic()


class Frame:
    def __init__(
        self, id: int, data_root: str, voxel_size: float = 1e-5, verbose: bool = False
    ):
        self.id = id
        self.verbose = verbose
        self.rgb_path = f"{data_root}/rgb/{id+1}.png"
        self.depth_path = f"{data_root}/depth/{id+1}.png"
        self.voxel_size = voxel_size
        self.rgbd_img, self.pcd = self.depth_image_to_point_cloud()
        self.pcd_down, self.pcd_fpfh = preprocess_point_cloud(
            self.pcd, self.voxel_size, self.verbose
        )

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
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        return rgbd_image, pcd
