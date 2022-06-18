import numpy as np
import cv2
import tensorflow as tf
import random
from gym.envs.mujoco.ant_goal_leg_no_reward import AntGoalLegEnv
from networks.skill2_learner8 import Skill2_Learner
from util import *
from modifyant import SetLeg



def GetColor(z1, z2):
    print(z1, z2)
    color =  ( int( np.clip((0.5 - 1.402  * z2) * 255, 0, 255 ) ),
                int( np.clip((0.5 -  0.344 * z1 - 0.714 * z2) * 255, 0, 255 )),
                int( np.clip((0.5 + 1.772 * z1) * 255, 0, 255) ))
    return color
                


gamma = 0.95
horizon = 100
test_num = 32

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
with sess.as_default():
    for leg in legparam:
        log_file = open(LOG_DIR + "log_zt.txt", "wt")
        log_file.write("z1\tz2\ttarget_x\ttarget_y\tfinal_x\tfinal_y\n")

        SetLeg(leg, "ant_noleg99.xml")
        envs = [ AntGoalLegEnv(99, [1., 1., 1., 1.]) for _ in range(test_num) ]
        for z1 in range(12):
            for z2 in range(12):
                color = GetColor( z1 / 12 - 0.5, z2 / 12 - 0.5 )
                print(color)
                if z1 <= 6 :
                    latents = np.array( [[z1 * 0.5 - 3.0, z2 * 0.5 - 3.0, -0.06 ]  for i in range(test_num)])
                else : 
                    latents = np.array( [[z1 * 0.7 - 4.2, z2 * 0.5 * (1.0 - (z1 - 6) * 0.1) - 3.0 * (1.0 - (z1 - 6) * 0.1), -0.06 ]  for i in range(test_num)])
                goal = learner.get_goal_batch(latents[:, :2], True)
                goal_swapped = np.array(np.swapaxes(goal,0,1))
                regulized_goal = goal_swapped[-1][0] / np.sqrt(np.sum(goal_swapped[-1][0] ** 2))


                states = []
                survive_vector = [True]  * test_num
                for i in range(test_num):
                    states.append(envs[i].reset())
                states = np.array(states)
                rewards = [0. for _ in range(test_num)]
                traj = np.zeros((traj_len, 2))

                for step in range(traj_len):
                    prev_states = states
                    #regulized_goal = [relative_goal[i] / np.sqrt(np.sum(relative_goal[i] ** 2)) for i in range(test_num)]

                    actions = learner.get_action_batch(states, latents, True)
                    actions = np.clip(actions, -1, 1)

                    states = []
                    for i in range(test_num):
                        state, reward, done, _ = envs[i].step(actions[i])
                        states.append(state)
                        
                        if z1 <= 6 :
                            traj[step] += state[:2]
                        else:
                            traj[step][0] += state[0] * 1.25
                            traj[step][1] += state[1]

                    states = np.array(states)
                print(z1, z2, regulized_goal, traj[-1])
                log_file.write(str(z1) + "\t" + str(z2) + "\t"  + str(regulized_goal[0]) + "\t" + str(regulized_goal[1]) 
                    + "\t" + str(traj[-1][0]) + "\t" + str(traj[-1][1]) + "\n" )


                cv2.polylines(real_traj_map, np.array([traj * 8. + 512.], np.int32), False,  color)
                cv2.imshow("real_traj_map", real_traj_map)

                cv2.waitKey(1)

    cv2.imwrite(LOG_DIR + "real_traj_map.png", real_traj_map)