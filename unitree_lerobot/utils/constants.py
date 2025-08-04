
import dataclasses
from typing import List, Dict

@dataclasses.dataclass(frozen=True)
class RobotConfig:
    motors: List[str]
    cameras: List[str]
    camera_to_image_key:Dict[str, str]
    json_state_data_name: List[str]
    json_action_data_name: List[str]
    #tactiles: List[str] = dataclasses.field(default_factory=list)


Z1_CONFIG = RobotConfig(
    motors=[
        "kLeftWaist",
        "kLeftShoulder",
        "kLeftElbow",
        "kLeftForearmRoll",
        "kLeftWristAngle",
        "kLeftWristRotate",
        "kLeftGripper",
        "kRightWaist",
        "kRightShoulder",
        "kRightElbow",
        "kRightForearmRoll",
        "kRightWristAngle",
        "kRightWristRotate",
        "kRightGripper",
    ],
    cameras=[
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_high', 'color_1': 'cam_left_wrist' ,'color_2': 'cam_right_wrist'},
    json_state_data_name = ['left_arm', 'right_arm'],
    json_action_data_name = ['left_arm', 'right_arm']
)


Z1_SINGLE_CONFIG = RobotConfig(
    motors=[
        "kWaist",
        "kShoulder",
        "kElbow",
        "kForearmRoll",
        "kWristAngle",
        "kWristRotate",
        "kGripper",
    ],
    cameras=[
        "cam_high",
        "cam_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_high', 'color_1': 'cam_wrist'},
    json_state_data_name = ['left_arm', 'right_arm'],
    json_action_data_name = ['left_arm', 'right_arm']
)


G1_GRIPPER_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftGripper",
        "kRightGripper",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
        ],
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_1':'cam_right_high', 'color_2': 'cam_left_wrist' ,'color_3': 'cam_right_wrist'},
    json_state_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand'],
    json_action_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
)


G1_DEX3_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftHandThumb0",
        "kLeftHandThumb1",
        "kLeftHandThumb2",
        "kLeftHandMiddle0",
        "kLeftHandMiddle1",
        "kLeftHandIndex0",
        "kLeftHandIndex1",
        "kRightHandThumb0",
        "kRightHandThumb1",
        "kRightHandThumb2",
        "kRightHandIndex0",
        "kRightHandIndex1",
        "kRightHandMiddle0",
        "kRightHandMiddle1",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_1':'cam_right_high', 'color_2': 'cam_left_wrist' ,'color_3': 'cam_right_wrist'},
    json_state_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand'],
    json_action_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
)

G1_INSPIRE_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kLeftHandPinky",
        "kLeftHandRing",
        "kLeftHandMiddle",
        "kLeftHandIndex",
        "kLeftHandThumbBend",
        "kLeftHandThumbRotation",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kRightHandPinky",
        "kRightHandRing",
        "kRightHandMiddle",
        "kRightHandIndex",
        "kRightHandThumbBend",
        "kRightHandThumbRotation",
    ],
    cameras=[
        "cam_left_high",
    ],
    camera_to_image_key = {'color_0': 'cam_left_high'},
    json_state_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand'],
    json_action_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
)

'''
G1_INSPIRE_TACTILE_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kLeftHandPinky",
        "kLeftHandRing",
        "kLeftHandMiddle",
        "kLeftHandIndex",
        "kLeftHandThumbBend",
        "kLeftHandThumbRotation",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kRightHandPinky",
        "kRightHandRing",
        "kRightHandMiddle",
        "kRightHandIndex",
        "kRightHandThumbBend",
        "kRightHandThumbRotation",
    ],
    cameras=[
        "cam_left_high",
    ],
    camera_to_image_key = {'color_0': 'cam_left_high'},
    json_state_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand'],
    json_action_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand'],
    tactiles=[
        'tactile.left_tactile.little_finger_tip',
        'tactile.left_tactile.little_finger_nail',
        'tactile.left_tactile.little_finger_pad',
        'tactile.left_tactile.ring_finger_tip',
        'tactile.left_tactile.ring_finger_nail',
        'tactile.left_tactile.ring_finger_pad',
        'tactile.left_tactile.middle_finger_tip',
        'tactile.left_tactile.middle_finger_nail',
        'tactile.left_tactile.middle_finger_pad',
        'tactile.left_tactile.index_finger_tip',
        'tactile.left_tactile.index_finger_nail',
        'tactile.left_tactile.index_finger_pad',
        'tactile.left_tactile.thumb_tip',
        'tactile.left_tactile.thumb_nail',
        'tactile.left_tactile.thumb_middle_section',
        'tactile.left_tactile.thumb_pad',
        'tactile.left_tactile.palm',
        'tactile.right_tactile.little_finger_tip',
        'tactile.right_tactile.little_finger_nail',
        'tactile.right_tactile.little_finger_pad',
        'tactile.right_tactile.ring_finger_tip',
        'tactile.right_tactile.ring_finger_nail',
        'tactile.right_tactile.ring_finger_pad',
        'tactile.right_tactile.middle_finger_tip',
        'tactile.right_tactile.middle_finger_nail',
        'tactile.right_tactile.middle_finger_pad',
        'tactile.right_tactile.index_finger_tip',
        'tactile.right_tactile.index_finger_nail',
        'tactile.right_tactile.index_finger_pad',
        'tactile.right_tactile.thumb_tip',
        'tactile.right_tactile.thumb_nail',
        'tactile.right_tactile.thumb_middle_section',
        'tactile.right_tactile.thumb_pad',
        'tactile.right_tactile.palm'
    ]
)
'''

ROBOT_CONFIGS = {
    "Unitree_Z1_Single": Z1_SINGLE_CONFIG,
    "Unitree_Z1_Dual": Z1_CONFIG,
    "Unitree_G1_Gripper": G1_GRIPPER_CONFIG,
    "Unitree_G1_Dex3": G1_DEX3_CONFIG,
    "Unitree_G1_Inspire": G1_INSPIRE_CONFIG,
    #"Unitree_G1_Tactile": G1_INSPIRE_TACTILE_CONFIG,
}