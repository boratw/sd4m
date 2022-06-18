import numpy as np
import cv2
import tensorflow as tf
import random
from gym.envs.mujoco.ant_goal_leg_no_reward import AntGoalLegEnv
from networks.skill2_learner8 import Skill2_Learner
from util import *
from modifyant import SetLeg



gamma = 0.95
horizon = 100
test_num = 16

env = AntGoalLegEnv(0, [1., 1., 1., 1.])
state_len = env.get_current_obs().size
action_len = env.action_space.shape[0]
qpos_len = env.model.nq
traj_len = 20
traj_track_len = 5
latent_len = 3
latent_body_len = 2
latent_preserve = 4
learner_batch = 32

no_action = np.zeros((action_len,), np.float32)


learner = Skill2_Learner(state_len, action_len, traj_len, traj_track_len, latent_len, latent_body_len, qpos_len, learner_batch,
    sampler_diverse=1., sampler_disperse=0., learner_lr=0.0001, sampler_lr=0.0001, learner_unity=0.1)



LOG_DIR = "data/skill2/test12_4/" + CreateLogPrefix()

latent_x = 2.0
latent_y = -0.6

sess = tf.Session()
learner_saver = tf.train.Saver(max_to_keep=0, var_list=learner.trainable_dict)
init = tf.global_variables_initializer()
sess.run(init)
learner_saver.restore(sess, "data/skill2/train12_2_2/2021-09-06_13-49-57_log1_learner_400.ckpt")
#learner_saver.restore(sess, "data/skill2/train12_4/2022-04-21_17-48-51_log1_learner_2400.ckpt")

legparam = [ 
    [1.0, 1.0, 1.0, 1.0]   ]

latent = np.random.normal(size=(test_num, latent_body_len))       
real_traj_map = np.full((1024, 1024, 3), 255, np.uint8)

res_result = [0.] * 72
with sess.as_default():
    for leg in legparam:
        log_file = open(LOG_DIR + "log_zt2.txt", "wt")
        log_file.write("z1\tz2\ttarget_x\ttarget_y\tfinal_x\tfinal_y\n")

        SetLeg(leg, "ant_noleg99.xml")
        envs = [ AntGoalLegEnv(99, [1., 1., 1., 1.]) for _ in range(test_num) ]
        for z1 in range(40):
            for z2 in range(40):
                latents = np.array( [[z1 * 0.2 - 3.0, z2 * 0.15 - 3.0, -0.06 ]  for i in range(test_num)])

                states = []
                survive_vector = [True]  * test_num
                for i in range(test_num):
                    states.append(envs[i].reset())
                states = np.array(states)
                rewards = [0. for _ in range(test_num)]
                traj = np.zeros((traj_len, 2))

                for step in range(horizon):
                    if step % traj_len == 0:
                        first_position = states[:, :2]
                    prev_states = states
                    #regulized_goal = [relative_goal[i] / np.sqrt(np.sum(relative_goal[i] ** 2)) for i in range(test_num)]

                    actions = learner.get_action_batch(np.concatenate([states[:, :2] - first_position, states[:, 2:]], axis=1), latents, True)
                    actions = np.clip(actions, -1, 1)

                    states = []
                    for i in range(test_num):
                        state, reward, done, _ = envs[i].step(actions[i])
                        states.append(state)

                    states = np.array(states)

                finalxy = np.sum(states[:, :2], axis=0)

                print(z1, z2, finalxy)
                if finalxy[0] > 0.:
                    finalxy[0] *= 1.25

                theta = np.arctan2(finalxy[1], finalxy[0])
                dist = np.sqrt(np.sum(finalxy ** 2))
                thetait = int(theta * 36 / np.pi) + 35
                if dist * 0.9 > res_result[thetait]:
                    res_result[thetait] = dist * 0.9
                if dist > res_result[(thetait + 1) % 72]:
                    res_result[(thetait + 1) % 72] = dist
                if dist * 0.9 > res_result[(thetait + 2) % 72]:
                    res_result[(thetait + 2) % 72] = dist * 0.9

        for i in range(72):
            log_file.write( str(i) + "\t" + str(res_result[i]) + "\n" )

            theta = i / 36 * np.pi - np.pi
            cv2.line(real_traj_map, (512, 512), (512 + int(np.cos(theta) * res_result[i] * 2.0) , 512 + int(np.sin(theta) * res_result[i] * 2.0)), (0, 0, 255))
            cv2.circle(real_traj_map, (512 + int(np.cos(theta) * res_result[i] * 2.0) , 512 + int(np.sin(theta) * res_result[i] * 2.0)), 8, (0, 0, 255))


    cv2.imwrite(LOG_DIR + "velocity.png", real_traj_map)