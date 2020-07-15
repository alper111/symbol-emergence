"""Reinforcement learning agent test script."""
import argparse
import yaml
import rospy
import torch
import torobo_wrapper
import models
import env

parser = argparse.ArgumentParser("Test PPO Agent.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

# READ SETTINGS
opts = yaml.safe_load(open(args.opts, "r"))
print(opts)

# INITIALIZE PPO AGENT
model = models.PPOAgent(state_dim=opts["state_dim"], hidden_dim=opts["hidden_dim"], action_dim=opts["action_dim"],
                        dist="gaussian", num_layers=opts["num_layers"], device=opts["device"], lr_p=opts["lr_p"],
                        lr_v=opts["lr_v"], K=opts["K"], batch_size=opts["batch_size"], eps=opts["eps"],
                        c_clip=opts["c_clip"], c_v=opts["c_v"], c_ent=opts["c_ent"])
model.load(opts["save"], ext="_last")

# INITIALIZE ROSNODE
rospy.init_node("test_node", anonymous=True)
rate = rospy.Rate(100)
rospy.sleep(1.0)
robot = torobo_wrapper.Torobo()
rospy.sleep(1.0)

# INITIALIZE ENVIRONMENT
world = env.Environment(robot=robot, objects=opts["objects"], rng_ranges=opts["ranges"])
rospy.sleep(0.5)
world.initialize()

print("="*10+"POLICY NETWORK"+"="*10)
print(model.policy)
print("="*10+"VALUE NETWORK"+"="*10)
print(model.value)
print("Training starts...")
reward_history = []

for epi in range(50):

    cumsum = 0.0
    obs = torch.tensor(world.reset(), dtype=torch.float, device=opts["device"])
    skip = False
    for t in range(opts["max_timesteps"]):
        with torch.no_grad():
            action, logprob = model.action(obs, std=False)
        obs, reward, done, success = world.step(obs.cpu().numpy(), action.cpu().numpy(), rate)
        obs = torch.tensor(obs, dtype=torch.float, device=opts["device"])
        if not success:
            skip = True
            break

        if done or (t == (opts["max_timesteps"]-1)):
            cumsum = reward
        else:
            cumsum = 0.0

    if t < 2:
        skip = True
    if skip:
        print("Breaking the episode")
        continue

    print("Reward: %.3f" % cumsum)

world.zerorobotpose()
