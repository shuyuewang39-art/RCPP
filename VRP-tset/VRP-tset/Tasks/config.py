import numpy as np
import json

def get_config():
    config_file_name = "./Tasks/config_example_01.json"  # a json configuration file
    json_file = open(config_file_name)  # Read the configuration from the json file
    config_data = json.load(json_file)

    H = config_data["UAV_DEFAULT"]["flight height"]  # 为稳定时的飞行高度
    alph = config_data["UAV_DEFAULT"]["horizontal angle of view"]  # 传感器的视野范围
    radius = config_data["UAV_DEFAULT"]["minimum turning radius"]  # 无人机的最小转弯半径
    start_position = config_data["UAV_DEFAULT"]["initial position"]  # 无人机的起始位置
    start_yaw = config_data["UAV_DEFAULT"]["initial yaw"]  # 无人机的起始位置
    curvature = 1/radius
    start_city = start_position.copy()
    lx = 2 * H * np.tan(alph/2)  # 探测范围
    start = [start_position[0], start_position[1], start_yaw]
    return start,lx,curvature