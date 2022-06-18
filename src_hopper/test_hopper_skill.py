import numpy as np
import cv2
import tensorflow as tf
import random
import sys
import argparse
from modifyhopper import SetBody
from gym.envs.mujoco.hopper_no_reward import HopperEnv
from networks.skill_learner import Skill_Learner
from networks.skill_follower_hopper import Skill_Follower
from util import *

def Main(args):
    gamma = 0.95
    horizon = 200
    env_num = 20
    task_num = 3
    goal_len = 1
    batch_num = 4

    envs = [HopperEnv() for _ in range(env_num)]
    state_len = envs[0].get_current_obs().size
    action_len = envs[0].action_space.shape[0]
    qpos_len = envs[0].model.nq
    traj_len = 10
    traj_track_len = 5
    latent_len = 3
    latent_body_len = 2
    latent_preserve = 4
    learner_batch = 32


    learner = Skill_Learner(state_len, action_len, traj_len, traj_track_len, latent_len, latent_body_len, qpos_len, task_num)
    follower = [Skill_Follower(state_len,  action_len, goal_len, str(i)) for i in range(task_num)]

    follower_savers = [tf.train.Saver(max_to_keep=0, var_list=follower[i].trainable_dict) for i in range(task_num)]



    LOG_DIR = args.log_dir + "/" + CreateLogPrefix()
    log_file = CreateLogFile(LOG_DIR + "log.txt")

    sess = tf.Session()
    log_file.write("Episode\tStep\tMove\tReward\n")

    learner_saver = tf.train.Saver(max_to_keep=0, var_list=learner.trainable_dict)

    with sess.as_default():

        learner.network_initialize()
        #follower_savers[2].restore(sess, "data/skill2/train9/2021-08-27_15-04-19_log1_follower2_600.ckpt")

        init = tf.global_variables_initializer()
        sess.run(init)

        #learner_saver.restore(sess, "logs/skill_hopper/2022-06-13_00-27-24_log1_learner_3900.ckpt")
        follower_savers[0].restore(sess, "logs/skill_hopper/2022-06-13_00-27-24_log1_follower0_3900.ckpt")
        follower_savers[1].restore(sess, "logs/skill_hopper/2022-06-13_00-27-24_log1_follower1_3900.ckpt")
        follower_savers[2].restore(sess, "logs/skill_hopper/2022-06-13_00-27-24_log1_follower2_3900.ckpt")

        #goals = np.random.uniform(0.5, 2.0, size=(batch_num, env_num, 1))
        goals = np.array([ [ [ i * 0.05 + 0.5] for i in range(env_num)] for _ in range(batch_num) ])
        for mass_it in range(21):
            mass = mass_it * 0.05 + 0.5
            mass_vector = [[mass]]  * env_num
            SetBody([mass], "hopper_modified.xml")
            envs = [HopperEnv(xml_file="hopper_modified.xml") for _ in range(env_num)]
            
            cur_step = 0.
            cur_move = 0.
            cur_reward = 0.
            cur_env_reward = np.zeros((env_num,))
            for latent in range(3):
                cur_latent_step = 0.
                cur_latent_move = 0.
                cur_latent_reward = 0.
                cur_latent_env_reward = np.zeros((env_num,))
                for batch in range(batch_num):
                    cur_batch_step = 0.
                    cur_batch_move = 0.
                    cur_batch_reward = 0.

                    
                    states = []
                    for i in range(env_num):
                        state = envs[i].reset()
                        states.append(state)
                    survive_vector = [True]  * env_num
                    states = np.array(states)
                    

                    for step in range(horizon):
                        #actions = learner.get_action_batch(states, new_latents, True)
                        actions = follower[latent].get_action_batch(states, goals[batch], False)
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

                                cur_latent_env_reward[i] += reward

                            if done:
                                survive_vector[i] = False

                            new_states.append(state)
                        states = np.array(new_states)
                        #envs[0].render()
                    
                
                    cur_batch_move /= cur_batch_step
                    cur_batch_reward /= cur_batch_step
                    cur_batch_step /= env_num 
                    print("Mass " + str(mass) + " Step " + str(cur_batch_step ) +  " Reward " +  str(cur_batch_reward ) + " Latent " + str(latent) )

                    cur_latent_move += cur_batch_move
                    cur_latent_step += cur_batch_step
                    cur_latent_reward += cur_batch_reward

                if cur_reward < cur_latent_reward:
                    cur_move = cur_latent_move
                    cur_step = cur_latent_step
                    cur_reward = cur_latent_reward
                    cur_env_reward = cur_latent_env_reward

            cur_move /= batch_num
            cur_reward /= batch_num
            cur_step /= batch_num
            cur_env_reward /= batch_num

            for i in range(env_num):
                log_file.write(str(mass) + "\t" + str(i * 0.05 + 0.5) + "\t" + str(cur_step) + "\t" + str(cur_env_reward[i]) + "\t")
                log_file.write("\n")
            log_file.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir')

    args = parser.parse_args()
    Main(args)