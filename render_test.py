import metaworld
import random
import matplotlib.pyplot as plt
# print(metaworld.ML1.ENV_NAMES)  # Check out the available environments
env_name='push-back-v2'
ml1 = metaworld.MT1(env_name) # Construct the benchmark, sampling tasks

env = ml1.train_classes[env_name]()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task
env._freeze_rand_vec = False
for i in range(10):
    obs = env.reset()  # Reset environment
    print(obs[-3:],env._freeze_rand_vec,env._last_rand_vec)
#image = env.sim.render(1024,1024)[::-1,:,:]
image=env.render( offscreen=True, camera_name="behindGripper", resolution=(640, 480))
plt.figure()
plt.imshow(image)
plt.show()
for i in range(100):
    a = env.action_space.sample()  # Sample an action
    obs, reward, done, info = env.step(a)
image=env.render( offscreen=True, camera_name="behindGripper", resolution=(640, 480))
plt.figure()
plt.imshow(image)