import time
import argparse
import csv
import os
import mujoco.viewer
import mujoco
import numpy as np
import torch
import pygame
from threading import Thread
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
import yaml
from collections import deque

cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)
x_vel_max, y_vel_max, yaw_vel_max = 2.0, 1.0, 1.0

joystick_use = True
joystick_opened = False

if joystick_use:

    pygame.init()

    try:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"cannot open joystick device:{e}")

    exit_flag = False

    def handle_joystick_input():
        global exit_flag, cmd
        
        
        while not exit_flag:
            pygame.event.get()

            cmd[0] = -joystick.get_axis(1) * x_vel_max
            cmd[1] = -joystick.get_axis(0) * y_vel_max
            cmd[2] = -joystick.get_axis(3) * yaw_vel_max

            pygame.time.delay(100)

    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input, daemon=True)
        joystick_thread.start()

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def load_config(args):
    """Load configuration from yaml file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", "r2_v1.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
        if args.policy is not None:
            policy_path = args.policy
        else:
            policy_path = config["policy_path"]

        if os.path.exists(policy_path):
            print(f"Using policy from: {policy_path}")
        else:
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

    return config, policy_path

def setup_joint_mappings(m, policy_joints):
    """Setup joint mappings between policy and xml"""
    policy_to_xml = []
    xml_joint_ids = []
    actuated_qpos_indices = []
    actuated_dof_indices = []
    ctrl_indices = []
    policy_joint_set = set(policy_joints)

    for i in range(1, m.njnt):
        jname = mujoco.mj_id2name(m, 3, i)
        if jname not in policy_joint_set:
            continue
        idx = policy_joints.index(jname)
        actuator_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        if actuator_id == -1:
            actuator_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{jname}_servo")
        if actuator_id == -1:
            raise ValueError(f"Policy joint '{jname}' does not have a matching actuator")

        policy_to_xml.append(idx)
        xml_joint_ids.append(i)
        actuated_qpos_indices.append(m.jnt_qposadr[i])
        actuated_dof_indices.append(m.jnt_dofadr[i])
        ctrl_indices.append(actuator_id)

    xml_to_policy = []
    for i in range(len(policy_to_xml)):
        idx = policy_to_xml.index(i)
        xml_to_policy.append(idx)

    missing_joints = [joint for joint in policy_joints if joint not in {mujoco.mj_id2name(m, 3, i) for i in xml_joint_ids}]
    if missing_joints:
        raise ValueError(f"Policy joints not found in model: {missing_joints}")

    return policy_to_xml, xml_to_policy, actuated_qpos_indices, actuated_dof_indices, ctrl_indices

def format_log_value(value, zero_epsilon=1e-12):
    """Format CSV values in a PlotJuggler-friendly fixed-point form."""
    value = float(value)
    if not np.isfinite(value):
        return "0.000000000"
    if abs(value) < zero_epsilon:
        value = 0.0
    return f"{value:.9f}"

def build_joint_log_row(log_time, qpos, qvel, tau, target_dof_pos):
    """Build one wide CSV row: time, all positions, all velocities, torques, targets."""
    values = (
        [log_time]
        + qpos.tolist()
        + qvel.tolist()
        + tau.tolist()
        + target_dof_pos.tolist()
    )
    return [format_log_value(value) for value in values]

def save_joint_log_csv(log_csv_path, joint_names, log_rows):
    """Write buffered joint logs to CSV."""
    if not log_rows:
        print("No joint log rows captured; CSV was not written.")
        return

    log_dir = os.path.dirname(os.path.abspath(log_csv_path))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    header = (
        ["time_s"]
        + [f"{joint_name}_pos" for joint_name in joint_names]
        + [f"{joint_name}_vel" for joint_name in joint_names]
        + [f"{joint_name}_tor" for joint_name in joint_names]
        + [f"{joint_name}_target_pos" for joint_name in joint_names]
    )
    expected_columns = len(header)
    valid_rows = [row for row in log_rows if len(row) == expected_columns]
    dropped_rows = len(log_rows) - len(valid_rows)
    if dropped_rows:
        print(f"Dropped {dropped_rows} malformed joint log rows before writing CSV.")
    if not valid_rows:
        print("No valid joint log rows captured; CSV was not written.")
        return

    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(valid_rows)

    last_time = float(valid_rows[-1][0])
    print(f"Saved joint log CSV: {log_csv_path}")
    print(f"Joint log samples: {len(valid_rows)}, columns: {len(header)}, duration: {last_time:.6f}s")

def build_policy_input(frame_stack, num_obs, num_actions, history_mode):
    """Build policy input from stacked single-frame observations."""
    stacked_obs = np.asarray(frame_stack, dtype=np.float32)
    if history_mode == "time":
        return stacked_obs.reshape(1, -1)
    if history_mode == "term":
        frames = stacked_obs.reshape(-1, num_obs)
        term_groups = [
            frames[:, 0:3].reshape(-1),
            frames[:, 3:6].reshape(-1),
            frames[:, 6:9].reshape(-1),
            frames[:, 9 : 9 + num_actions].reshape(-1),
            frames[:, 9 + num_actions : 9 + 2 * num_actions].reshape(-1),
            frames[:, 9 + 2 * num_actions : 9 + 3 * num_actions].reshape(-1),
        ]
        return np.concatenate(term_groups, axis=0, dtype=np.float32).reshape(1, -1)
    raise ValueError(f"Unsupported history mode: {history_mode}")

def run_simulation(policy_path, config, log_csv_path, log_duration, history_mode):
    """
    Run the Mujoco simulation using the provided policy and configuration.
    
    Args:
        policy_path: Path to the trained policy file
        config: Configuration dictionary containing simulation settings
    """
    # Extract configuration parameters
    xml_path = config["xml_path"]
    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    policy_joints = config["policy_joints"]
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    base_init_pos = np.array(config.get("base_init_pos", [0.0, 0.0, 0.67]), dtype=np.float32)
    
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    obs_history_length = config.get("obs_history_length", 10)
    
    # Initialize context variables
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    
    # Load robot model
    print("Loading robot model...")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    (
        policy_to_xml,
        xml_to_policy,
        actuated_qpos_indices,
        actuated_dof_indices,
        ctrl_indices,
    ) = setup_joint_mappings(m, policy_joints)
    joint_names_xml = [policy_joints[idx] for idx in policy_to_xml]

    default_angles = default_angles[policy_to_xml]
    if not (len(kps) == len(kds) == len(default_angles) == len(ctrl_indices) == num_actions):
        raise ValueError(
            "Config/action size mismatch: "
            f"kps={len(kps)}, kds={len(kds)}, default_angles={len(default_angles)}, "
            f"controlled_joints={len(ctrl_indices)}, num_actions={num_actions}"
        )
    ctrl_range = m.actuator_ctrlrange[ctrl_indices]
    ctrl_min = ctrl_range[:, 0]
    ctrl_max = ctrl_range[:, 1]

    print("Policy to XML joint mapping:", policy_to_xml)
    print("XML to Policy joint mapping:", xml_to_policy)
    print("Controlled joints:", joint_names_xml)
    print("History mode:", history_mode)
    target_dof_pos = default_angles.copy()
    
    # Set initial joint positions to default_angles
    d.qpos[0:3] = base_init_pos
    d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    d.qpos[actuated_qpos_indices] = default_angles.copy()
    mujoco.mj_forward(m, d)  # Update the simulation state
    
    # Initialize frame stack
    frame_stack = deque(maxlen=obs_history_length)
    for _ in range(obs_history_length):
        frame_stack.append(obs.copy())
    # Time sync configuration
    viewer_sync_interval = 2
    print_interval = 2.0  # 每2秒打印一次时间同步状态
    # Load policy
    print("Loading policy...")
    policy = torch.jit.load(policy_path)

    log_rows = []
    log_start_time = d.time
    
    print("Starting simulation...")
    try:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            # 改进的实时同步逻辑
            sim_start_time = time.time()
            sim_time = 0.0  # 仿真时间
            last_print_time = sim_start_time
            while viewer.is_running() and time.time() - sim_start_time < simulation_duration:
                step_start = time.time()
                # 计算应该执行的仿真时间与实际时间的对齐
                wall_time_elapsed = time.time() - sim_start_time
                
                # PD control
                qpos = d.qpos[actuated_qpos_indices].copy()
                qvel = d.qvel[actuated_dof_indices].copy()
                tau = pd_control(target_dof_pos, qpos, kps, np.zeros_like(kds), qvel, kds)
                tau = np.clip(tau, ctrl_min, ctrl_max)
                log_time = d.time - log_start_time
                if log_time <= log_duration:
                    log_rows.append(build_joint_log_row(log_time, qpos, qvel, tau, target_dof_pos))
                d.ctrl[ctrl_indices] = tau
                
                # Step physics
                mujoco.mj_step(m, d)
                counter += 1
                # 更新仿真时间
                sim_time = d.time     
                # Policy inference at lower frequency
                if counter % control_decimation == 0:
                    # Create observation
                    qj = d.qpos[actuated_qpos_indices]
                    dqj = d.qvel[actuated_dof_indices]
                    quat = d.qpos[3:7]
                    omega = d.qvel[3:6]
                    
                    qj = (qj - default_angles) * dof_pos_scale
                    dqj = dqj * dof_vel_scale
                    gravity_orientation = get_gravity_orientation(quat)
                    omega = omega * ang_vel_scale
                    
                    obs[:3] = omega
                    obs[3:6] = gravity_orientation
                    obs[6:9] = cmd * cmd_scale
                    obs[9 : 9 + num_actions] = qj[xml_to_policy]
                    obs[9 + num_actions : 9 + 2 * num_actions] = dqj[xml_to_policy]
                    obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action[xml_to_policy]
                    
                    frame_stack.append(obs.copy())
                    policy_input = build_policy_input(frame_stack, num_obs, num_actions, history_mode)
                    obs_tensor = torch.from_numpy(policy_input)
                    
                    # Policy inference
                    action = policy(obs_tensor).detach().numpy().squeeze()
                    action = action[policy_to_xml]
                    # action = np.zeros_like(action)  # 临时屏蔽动作输出，保持默认姿态
                    
                    # Transform action to target_dof_pos
                    target_dof_pos = action * action_scale + default_angles
                    # target_dof_pos = default_angles
                # Update viewer
                # Sync viewer
                # 优化：减少viewer同步频率以提升性能
                if counter % viewer_sync_interval == 0:
                    viewer.sync()
                
                # 改进的时间同步：精确对齐wall-time
                # 如果仿真进度落后，跳过sleep直接进行下一步
                # 如果仿真进度超前，休眠以保持同步
                time_elapsed_this_step = time.time() - step_start
                target_step_time = m.opt.timestep
                
                # 计算需要休眠的时间
                sleep_time = target_step_time - time_elapsed_this_step
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # 如果sleep_time <= 0，表示仿真跟不上实时，此时不sleep，继续运行
                
                # 定期打印时间同步状态
                if time.time() - last_print_time > print_interval:
                    wall_elapsed = time.time() - sim_start_time
                    time_ratio = sim_time / wall_elapsed if wall_elapsed > 0 else 0
                    print(f"[TIME SYNC] Wall time: {wall_elapsed:.2f}s | Sim time: {sim_time:.2f}s | Ratio: {time_ratio:.3f}x | Steps: {counter}")
                    last_print_time = time.time()
    finally:
        #暂时先关闭这个功能
        save_joint_log_csv(log_csv_path, joint_names_xml, log_rows)
        pass
    
    print("Simulation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='rw_25dof deploy mujoco')
    parser.add_argument('--policy', type=str, default="checkpoint/policy.pt",
                    help='Direct path to policy file (overrides config file)')
    parser.add_argument('--log-csv', type=str, default="deploy_mujoco_logs/joint_log.csv",
                    help='Path to save joint log CSV')
    parser.add_argument('--log-duration', type=float, default=10.0,
                    help='Simulation-time duration to record joint logs, in seconds')
    history_mode_group = parser.add_mutually_exclusive_group()
    history_mode_group.add_argument('--term', dest='history_mode', action='store_const', const='term',
                    help='Pack observation history by term groups (default)')
    history_mode_group.add_argument('--time', dest='history_mode', action='store_const', const='time',
                    help='Pack observation history frame-by-frame in chronological order')
    parser.set_defaults(history_mode='term')
    
    args = parser.parse_args()
    
    # Load configuration
    config, policy_path = load_config(args)
    
    try:
        # Run simulation in main thread
        run_simulation(policy_path, config, args.log_csv, args.log_duration, args.history_mode)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        # Clean up
        if joystick_use and joystick_opened:
            exit_flag = True
            print("Cleaning up joystick thread...")
        if joystick_use:
            pygame.quit()
        print("Exit complete")
