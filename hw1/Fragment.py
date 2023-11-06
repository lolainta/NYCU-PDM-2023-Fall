from Frame import Frame
from Reconstructor import Reconstructor
from utils import preprocess_point_cloud, pairwise_registration


class Fragment:
    def __init__(self, id, sid, eid):
        self.id = id
        self.sid = sid
        self.eid = eid
        self.n_frames = eid - sid

    def merge_from_frame(self, data: str, voxel_size: float, verb: bool):
        frames = [Frame(i, data, voxel_size, verb) for i in range(self.sid, self.eid)]
        r = Reconstructor(self.id, self.sid, self.eid, frames, pairwise_registration)
        self.pcd, self.cam_pose = r.reconstruct()
        return self.pcd, self.cam_pose

    def set_data(self, s, voxel_size, verbose):
        self.pcd = s[0]
        self.pcd_down, self.pcd_fpfh = preprocess_point_cloud(
            self.pcd, voxel_size, verbose
        )
        self.cam_pose = s[1]
