import numpy as np
import cv2
import tensorflow as tf
import random
import sys
import argparse
from modifyhopper import SetBody
from gym.envs.mujoco.hopper_no_reward import HopperEnv
from networks.explorer_trpo_epopt import Explorer_TRPO
from util import *

def Main(args):
    gamma = 0.95
    horizon = 200
    env_num = 20

    envs = [HopperEnv() for _ in range(env_num)]
    state_len = envs[0].get_current_obs().size
    action_len = envs[0].action_space.shape[0]

    follower = Explorer_TRPO(state_len, action_len, 1, env_num=env_num )





    LOG_DIR = args.log_dir + "/" + CreateLogPrefix()
    log_file = CreateLogFile(LOG_DIR + "log.txt")

    sess = tf.Session()
    follower_saver = tf.train.Saver(max_to_keep=0, var_list=follower.trainable_dict)

    log_file.write("Episode\tStep\tMove\tReward")
    log_file.write(follower.log_caption())
    log_file.write("\n")


    with sess.as_default():

        follower.network_initialize()
        #follower_savers[2].restore(sess, "data/skill2/train9/2021-08-27_15-04-19_log1_follower2_600.ckpt")

        init = tf.global_variables_initializer()
        sess.run(init)

        follower_saver.restore(sess, "logs/trpo_hopper_epopt3/2022-06-16_17-48-42_log1_follower_4000.ckpt")

        goals = np.random.uniform(0.5, 2.0, size=(10, env_num, 1))
        for mass_it in range(21):
            mass = mass_it * 0.05 + 0.5
            mass_vector = [[mass]]  * env_num
            print(mass_vector)
            SetBody([mass], "hopper_modified.xml")
            envs = [HopperEnv(xml_file="hopper_modified.xml") for _ in range(env_num)]
            cur_step = 0.
            cur_move = 0.
            cur_reward = 0.
        
            for batch in range(10):
                cur_batch_step = 0.
                cur_batch_move = 0.
                cur_batch_reward = 0.

                                    
                
                states = []
                for i in range(env_num):
                    state = envs[i].reset()
                    states.append(state)
                survive_vector = [True]  * env_num
                states = np.array(states)

                env_reward = [0.]  * env_num
                env_step = [0.]  * env_num
                for step in range(horizon):
                    actions = follower.get_action_batch(states, goals[batch], mass_vector, record=False)
                    new_states = []
                    reward_set = []
                    for i in range(env_num):
                        state, reward, done, info = envs[i].step(actions[i])
                        reward += 0.5 - np.abs(info['x_velocity'] - goals[batch][i][0])
                        #if i == 0:
                        #    print(info['x_velocity'], goals[batch][i][0])
                        #vel_reward = 2. /  (1. + np.abs(info['x_velocity'] - goals[batch][i][0]))
                        #reward += (vel_reward if vel_reward < 1. else 1.)
                        reward_set.append(reward)
                        if survive_vector[i]:
                            cur_batch_step += 1
                            cur_batch_reward += reward

                            env_step[i] += 1.
                            env_reward[i] += reward

                        if done:
                            survive_vector[i] = False

                        new_states.append(state)
                    states = np.array(new_states)
                    envs[0].render()
                      
                follower.episode_finished()
               
                cur_batch_move /= cur_batch_step
                cur_batch_reward /= cur_batch_step
                cur_batch_step /= env_num 
                print("Mass " + str(mass) + " Step " + str(cur_batch_step ) +  " Reward " +  str(cur_batch_reward ) )

                cur_move += cur_batch_move
                cur_step += cur_batch_step
                cur_reward += cur_batch_reward

            cur_move /= 10
            cur_reward /= 10
            cur_step /= 10

            log_file.write(str(mass) + "\t" + str(cur_step) + "\t" + str(cur_reward) + "\t")
            log_file.write("\n")
            log_file.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir')

    args = parser.parse_args()
    Main(args)