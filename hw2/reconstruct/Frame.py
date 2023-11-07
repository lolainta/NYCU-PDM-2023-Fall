import open3d as o3d
import numpy as np
from enum import Enum, auto


def get_intrinsic():
    width, height = 512, 512
    fov = 90
    fx = width / (2 * np.tan(np.deg2rad(fov / 2)))
    fy = height / (2 * np.tan(np.deg2rad(fov / 2)))
    cx = width / 2
    cy = height / 2
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


intrinsic = get_intrinsic()


class BaseFrameType(Enum):
    RGB = auto()
    SEMENTIC = auto()


class Frame:
    def __init__(
        self,
        id: int,
        data_root: str,
        img_type: BaseFrameType,
        voxel_down,
        voxel_size: float = 1e-5,
        verbose: bool = False,
    ):
        self.id = id
        self.verbose = verbose
        self.rgb_path = f"{data_root}/rgb/{id+1}.png"
        self.sementic_path = f"{data_root}/semantic/{id+1}.png"
        if img_type == BaseFrameType.SEMENTIC:
            self.img_path = self.sementic_path
        elif img_type == BaseFrameType.RGB:
            self.img_path = self.rgb_path
        self.depth_path = f"{data_root}/depth/{id+1}.png"
        self.voxel_size = voxel_size
        self.rgbd_img, self.pcd = self.depth_image_to_point_cloud()
        self.pcd_down, self.pcd_fpfh = voxel_down(
            self.pcd, self.voxel_size, self.verbose
        )

    def depth_image_to_point_cloud(self):
        color_raw = o3d.io.read_image(self.img_path)
        depth_raw = o3d.io.read_image(self.depth_path)
        image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw,
            depth_raw,
            convert_rgb_to_intensity=False,
            depth_scale=0.1,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image, intrinsic)
        return image, pcd
