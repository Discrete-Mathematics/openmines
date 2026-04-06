import gymnasium as gym

# Create environment
env = gym.make('mine/Mine-v1-dense', config_file="./conf/north_pit_mine.json")  # or mine/Mine-v1-sparse

# Reset environment
obs, info = env.reset()

# Run an episode
for _ in range(1000):
    # Execute using suggested action
    obs, reward, done, truncated, info = \
       env.step(info["sug_action"])
    if done or truncated:
        break

env.close()
