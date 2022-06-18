import numpy as np
import cv2
import tensorflow as tf
import random
import sys
import argparse
from gym.envs.mujoco.hopper_no_reward import HopperEnv
from networks.explorer_trpo import Explorer_TRPO
from util import *

def Main(args):
    gamma = 0.96
    horizon = 200
    env_num = 16

    envs = [HopperEnv() for _ in range(env_num)]
    state_len = envs[0].get_current_obs().size
    action_len = envs[0].action_space.shape[0]

    follower = Explorer_TRPO(state_len, action_len, 1, env_num=env_num, policy_gamma=0.98)


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

        #follower_saver.restore(sess, "logs/trpo_hopper/2022-06-07_18-36-35_log1_follower_1000.ckpt")
        for episode in range(1, 2001):

            cur_step = 0.
            cur_diff = 0.
            cur_reward = 0.
        
            for batch in range(10):
                cur_batch_step = 0.
                cur_batch_diff = 0.
                cur_batch_reward = 0.

                                    
                
                states = []
                for i in range(env_num):
                    state = envs[i].reset()
                    states.append(state)
                
                #prev_vel = [0.] * env_num
                survive_vector = [True]  * env_num
                states = np.array(states)
                goals = np.random.uniform(0.5, 2.0, size=(env_num, 1))

                for step in range(horizon):
                    if any(survive_vector) == False:
                        break
                    actions = follower.get_action_batch(states, goals, survive_vector)
                    new_states = []
                    reward_set = []
                    new_survive_vector = survive_vector[:]
                    for i in range(env_num):
                        if survive_vector[i]:
                            state, reward, done, info = envs[i].step(actions[i])
                            #reward += (np.abs(prev_vel[i] - goals[i][0]) - np.abs(info['x_velocity'] - goals[i][0])) * 10.
                            #vel_reward = 1. -  (np.abs(info['x_velocity'] - goals[i][0]) * 0.2)
                            #reward += 0.5 - np.abs(info['x_velocity'] - goals[i][0]) * 0.1
                            if info['x_velocity'] < goals[i][0] :
                                reward += info['x_velocity'] * 0.5
                            elif info['x_velocity'] < goals[i][0] * 3. :
                                reward += (goals[i][0] * 0.75 - info['x_velocity'] * 0.25) 
                        
                            cur_batch_step += 1
                            cur_batch_diff += np.abs(info['x_velocity'] - goals[i][0])
                            cur_batch_reward += reward

                            if done:
                                #reward -= 10.
                                new_survive_vector[i] = False
                            else:
                                reward += 0.5
                            reward_set.append(reward)
                            new_states.append(state)
                            #prev_vel[i] = info['x_velocity']
                        else:
                            reward_set.append(0.)
                            new_states.append(states[i])
                            
                    states = np.array(new_states)
                    follower.push_reward_batch(reward_set, survive_vector)
                    survive_vector = new_survive_vector
                                  
                cur_batch_diff /= cur_batch_step
                cur_batch_reward /= cur_batch_step
                cur_batch_step /= env_num          
                print("Episode " + str(episode) + " Step " + str(cur_batch_step )+  " Move " +  str(cur_batch_diff )
                    +  " Reward " +  str(cur_batch_reward ) )

                cur_diff += cur_batch_diff
                cur_step += cur_batch_step
                cur_reward += cur_batch_reward

                follower.episode_finished()

            cur_diff /= batch
            cur_step /= batch
            cur_reward /= batch

            follower.optimize_batch()
            follower.log_print()

            log_file.write(str(episode) + "\t")
            log_file.write(str(cur_step) + "\t" + str(cur_diff)  + "\t" + str(cur_reward) + "\t")
            log_file.write(follower.current_log())
            log_file.write("\n")
            log_file.flush()

            #print(follower.scaler.means)
            #print(follower.scaler.vars)

            if episode % 100 == 0:
                follower_saver.save(sess, LOG_DIR + "log1_follower_" + str(episode) + ".ckpt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir')

    args = parser.parse_args()
    Main(args)