import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import json
from icecream import ic
from copy import deepcopy
import numpy as np
from enums import Objects, ColorTable


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img


def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def quat2rot(agent_state):
    res = euler_from_quaternion(
        agent_state.rotation.x,
        agent_state.rotation.y,
        agent_state.rotation.z,
        agent_state.rotation.w,
    )
    assert np.equal(res[0], res[2]), f"{res[0]} {res[2]}"
    if res[1] < 0:
        if np.equal(res[0], 0):
            ans = -res[1]
        else:
            ans = np.pi + res[1]
    else:
        if np.equal(res[0], 0):
            ans = res[1]
        else:
            ans = -np.pi + res[1]
    ret = np.rad2deg(ans) - 90
    if -270 < ret < -180:
        ret += 360
    ic(res, ans, ret)
    return ret


class Navigator:
    def make_simple_cfg(settings):
        # simulator backend
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = settings["scene"]

        # In the 1st example, we attach only one sensor,
        # a RGB visual sensor, to the agent
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
        rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        rgb_sensor_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        # depth snesor
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [settings["height"], settings["width"]]
        depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        depth_sensor_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        # semantic snesor
        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
        semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        semantic_sensor_spec.orientation = [
            settings["sensor_pitch"],
            0.0,
            0.0,
        ]
        semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        # agent
        agent_cfg = habitat_sim.agent.AgentConfiguration()

        agent_cfg.sensor_specifications = [
            rgb_sensor_spec,
            depth_sensor_spec,
            semantic_sensor_spec,
        ]
        ##################################################################
        ### change the move_forward length or rotate angle
        ##################################################################
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(amount=0.01),  # 0.01 means 0.01 m
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left",
                habitat_sim.agent.ActuationSpec(amount=1.0),  # 1.0 means 1 degree
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=1.0)
            ),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def goto(self, target):
        target = deepcopy(target)
        agent_state = self.agent.get_state()
        ic(agent_state.position)
        ic(agent_state.rotation)
        ic(target)
        path = target - agent_state.position
        # ic(path)
        rho = np.sqrt(path[0] ** 2 + path[2] ** 2)
        phi = np.arctan2(path[2], path[0])
        # ic(rho, phi)
        cur_rot = quat2rot(agent_state)
        # ic(cur_rot, np.rad2deg(phi))
        n_rotate = np.rad2deg(phi) - cur_rot
        ic(n_rotate)
        while abs(n_rotate) > 1:
            if n_rotate > 0:
                obs = self.sim.step("turn_right")
                n_rotate -= 1
            else:
                obs = self.sim.step("turn_left")
                n_rotate += 1
            self.show(obs)
        ic(quat2rot(self.agent.get_state()))
        ic(self.agent.get_state().position)
        n_forward = rho / 0.01
        ic(n_forward)
        while abs(n_forward) > 1:
            n_forward -= 1
            obs = self.sim.step("move_forward")
            self.show(obs)
        ic(self.agent.get_state().position)
        diff = self.agent.get_state().position - agent_state.position
        ic(path, diff)

    def show(self, obs):
        org = transform_rgb_bgr(obs["color_sensor"])
        sementic = self.id_to_label[obs["semantic_sensor"]]
        # ic(sementic)
        sem_id = ColorTable[tuple(self.destination)]
        mask = np.zeros((512, 512, 3))
        mask[sementic == sem_id] = np.array([0, 0, 255])

        mask = mask.astype(np.uint8)
        # ic(mask.shape, org.shape)
        org = cv2.addWeighted(mask, 0.5, org, 0.5, 0)
        cv2.imshow("RGB", org)

        self.out.write(org)
        cv2.waitKey(1)

    def done(self):
        self.video.release()
        cv2.destroyAllWindows()

    def navigateAndSee(self, action):
        if action in self.action_names:
            observations = self.sim.step(action)
            # print("action: ", action)

            cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
            # cv2.imshow("depth", transform_depth(observations["depth_sensor"]))
            cv2.imshow(
                "semantic",
                transform_semantic(self.id_to_label[observations["semantic_sensor"]]),
            )
            agent_state = self.agent.get_state()
            sensor_state = agent_state.sensor_states["color_sensor"]
            print("camera pose: x y z rw rx ry rz")
            print(
                sensor_state.position[0],
                sensor_state.position[1],
                sensor_state.position[2],
                sensor_state.rotation.w,
                sensor_state.rotation.x,
                sensor_state.rotation.y,
                sensor_state.rotation.z,
            )
            print(f"current rotation: {quat2rot(agent_state)}")

    def interactive(self):
        print("#############################")
        print("use keyboard to control the agent")
        print(" w for go forward  ")
        print(" a for turn left  ")
        print(" d for trun right  ")
        print(" f for finish and quit the program")
        print("#############################")
        action = "move_forward"
        self.navigateAndSee(action)

        while True:
            keystroke = cv2.waitKey(0)
            if keystroke == ord(self.FORWARD_KEY):
                action = "move_forward"
                self.navigateAndSee(action)
                print("action: FORWARD")
            elif keystroke == ord(self.LEFT_KEY):
                action = "turn_left"
                self.navigateAndSee(action)
                print("action: LEFT")
            elif keystroke == ord(self.RIGHT_KEY):
                action = "turn_right"
                self.navigateAndSee(action)
                print("action: RIGHT")
            elif keystroke == ord(self.FINISH):
                print("action: FINISH")
                break
            else:
                print("INVALID KEY")
                continue

    # This is the scene we are going to load.
    # support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
    ### put your scene path ###
    test_scene = "./replica_v1/apartment_0/habitat/mesh_semantic.ply"
    path = "./replica_v1/apartment_0/habitat/info_semantic.json"

    # global test_pic
    #### instance id to semantic id
    with open(path, "r") as f:
        annotations = json.load(f)

    id_to_label = []
    instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
    for i in instance_id_to_semantic_label_id:
        if i < 0:
            id_to_label.append(0)
        else:
            id_to_label.append(i)
    id_to_label = np.asarray(id_to_label)

    ######

    sim_settings = {
        "scene": test_scene,  # Scene path
        "default_agent": 0,  # Index of the default agent
        "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
        "width": 512,  # Spatial resolution of the observations
        "height": 512,
        "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
    }

    # This function generates a config for the simulator.
    # It contains two parts:
    # one for the simulator backend
    # one for the agent, where you can attach a bunch of sensors

    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # obtain the default, discrete actions that an agent can perform
    # default action space contains 3 actions: move_forward, turn_left, and turn_right
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    print("Discrete action space: ", action_names)

    def __init__(self, init_pos: np.array, destination: Objects):
        self.tar = destination
        self.destination = np.array(destination.value)
        # initialize an agent
        self.agent = self.sim.initialize_agent(self.sim_settings["default_agent"])

        # Set agent state
        agent_state = habitat_sim.AgentState()
        agent_state.position = init_pos  # agent in world space
        self.agent.set_state(agent_state)

        self.video = cv2.VideoCapture(0)
        self.out = cv2.VideoWriter(
            f"results/{self.tar.name}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            (512, 512),
        )

    FORWARD_KEY = "w"
    LEFT_KEY = "a"
    RIGHT_KEY = "d"
    FINISH = "f"
