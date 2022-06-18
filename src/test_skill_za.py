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
test_num = 64

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
learner_saver.restore(sess, "data/skill2/train12_4/2022-04-21_17-48-51_log1_learner_2400.ckpt")
#learner_saver.restore(sess, "data/skill2/train12_2_1/2021-09-06_10-18-51_log1_learner_900.ckpt")

legparam = [ 
    [1.0, 1.0, 1.0, 1.0], 
    [0.01, 1.0, 1.0, 1.0], 
    [1.0, 0.01, 1.0, 1.0],
    [0.5, 1.0, 1.0, 1.0],
    [1.5, 1.0, 1.0, 1.0],
    [1.0, 1.0, 0.01, 1.0]]
    #[0.5, 0.5, 1.0, 1.0]]
    #[1.0, 1.0, 0.01, 1.0]]
    #[1.0, 1.0, 1.5, 1.0],
    #[0.5, 0.5, 1.0, 1.0],
    #[1.0, 0.5, 0.5, 1.0],    ]

latent = np.random.normal(size=(test_num, latent_body_len))       
with sess.as_default():
    for leg in legparam:
        log_file = open(LOG_DIR + "log_" + str(leg[0]) + "_" + str(leg[1]) + "_" + str(leg[2]) + "_" + str(leg[3]) + ".txt", "wt")
        log_file.write("latent0\tlatent1\tlatent2\tfinal_pos_x\tfinal_pos_y\treward\n")

        SetLeg(leg, "ant_noleg99.xml")
        envs = [ AntGoalLegEnv(99, [1., 1., 1., 1.]) for _ in range(test_num) ]
        for z1 in range(301):
            latent_z = z1 * 0.02 - 2.0   
            latents = np.array( [[latent[i][0], latent[i][1], latent_z ]  for i in range(test_num)])
            goal = learner.get_goal_batch(latent, True)
            goal_swapped = np.array(np.swapaxes(goal,0,1))
            regulized_goal = [goal_swapped[-1][i] / np.sqrt(np.sum(goal_swapped[-1][i] ** 2)) for i in range(test_num)]
            states = []
            survive_vector = [True]  * test_num
            for i in range(test_num):
                states.append(envs[i].reset())
            states = np.array(states)
            rewards = [0. for _ in range(test_num)]
            for step in range(horizon):
                if step % traj_len == 0:
                    relative_goal = np.zeros((test_num, 2), np.float32)
                    first_position = states[:, :2]
                if step % traj_track_len == 0:
                    relative_goal += goal_swapped[(step % traj_len) // traj_track_len]

                prev_states = states
                states_noxy = states[:, 2:]
                #regulized_goal = [relative_goal[i] / np.sqrt(np.sum(relative_goal[i] ** 2)) for i in range(test_num)]

                actions = learner.get_action_batch(np.concatenate([states[:, :2] - first_position, states_noxy], axis=1), latents, True)
                actions = np.clip(actions, -1, 1)

                states = []
                for i in range(test_num):
                    if survive_vector[i]:
                        state, reward, done, _ = envs[i].step(actions[i])
                        states.append(state)
                        body_difference = state[:2] - prev_states[i][:2]
                        reward += np.dot(body_difference, regulized_goal[i]) * 20.
                        rewards[i] += reward
                        if done:
                            survive_vector[i] = False
                        relative_goal[i] -= body_difference
                    else:
                        states.append(prev_states[i])
                states = np.array(states)
            rewards = np.array(rewards)
            print(leg, latent_z, np.mean(rewards), np.std(rewards))
            log_file.write(str(latent_z) + "\t"  + str(np.mean(rewards)) + "\t"  + str(np.std(rewards)) + "\n" )




