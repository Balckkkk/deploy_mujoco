import time
import argparse
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

cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
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
    config_path = os.path.join(script_dir, "config", "r2_v2.yaml")
    
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
    for i in range(1, m.njnt):
        jname = mujoco.mj_id2name(m, 3, i)
        idx = policy_joints.index(jname)
        policy_to_xml.append(idx)

    xml_to_policy = []
    for i in range(len(policy_to_xml)):
        idx = policy_to_xml.index(i)
        xml_to_policy.append(idx)
    
    return policy_to_xml, xml_to_policy

def run_simulation(policy_path, config):
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
    
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    
    # Initialize context variables
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    
    # Load robot model
    print("Loading robot model...")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    policy_to_xml = []
    for i in range(1, m.njnt):
        jname = mujoco.mj_id2name(m, 3, i)
        idx = policy_joints.index(jname)
        policy_to_xml.append(idx)

    xml_to_policy = []
    for i in range(len(policy_to_xml)):
        idx = policy_to_xml.index(i)
        xml_to_policy.append(idx)

    default_angles = default_angles[policy_to_xml]
    target_dof_pos = default_angles.copy()
    
    # Set initial joint positions to default_angles
    d.qpos[7:] = default_angles.copy()
    mujoco.mj_forward(m, d)  # Update the simulation state
    
    # Initialize frame stack
    frame_stack = deque(maxlen=5)
    for _ in range(5):
        frame_stack.append(obs.copy())
        mujoco.mj_step(m, d)
    
    # Load policy
    print("Loading policy...")
    policy = torch.jit.load(policy_path)
    
    print("Starting simulation...")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # PD control
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            
            # Step physics
            mujoco.mj_step(m, d)
            counter += 1
            
            # Policy inference at lower frequency
            if counter % control_decimation == 0:
                # Create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
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
                stacked_obs = np.concatenate(frame_stack, axis=0)
                
                # Reshape observations for policy
                obs_omega = np.asarray(stacked_obs).reshape(5, num_obs)[:, 0:3].reshape(-1)
                obs_gravity_orientation = np.asarray(stacked_obs).reshape(5, num_obs)[:, 3:6].reshape(-1)
                obs_cmd = np.asarray(stacked_obs).reshape(5, num_obs)[:, 6:9].reshape(-1)
                obs_pos = np.asarray(stacked_obs).reshape(5, num_obs)[:, 9:9 + num_actions].reshape(-1)
                obs_vel = np.asarray(stacked_obs).reshape(5, num_obs)[:, 9 + num_actions : 9 + 2 * num_actions].reshape(-1)
                obs_action = np.asarray(stacked_obs).reshape(5, num_obs)[:, 9 + 2 * num_actions : 9 + 3 * num_actions].reshape(-1)
                
                big_group_major = np.concatenate([
                    obs_omega,
                    obs_gravity_orientation,
                    obs_cmd,
                    obs_pos,
                    obs_vel,
                    obs_action,
                ], axis=0)
                obs_tensor = torch.from_numpy(big_group_major).unsqueeze(0)
                
                # Policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                action = action[policy_to_xml]
                
                # Transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles
            
            # Sync viewer
            viewer.sync()
            
            # Time keeping
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    print("Simulation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='rw_25dof deploy mujoco')
    parser.add_argument('--policy', type=str, default="checkpoint/policy.pt",
                    help='Direct path to policy file (overrides config file)')
    
    args = parser.parse_args()
    
    # Load configuration
    config, policy_path = load_config(args)
    
    try:
        # Run simulation in main thread
        run_simulation(policy_path, config)
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
