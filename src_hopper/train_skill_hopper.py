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
    env_num = 16
    task_num = 3
    goal_len = 1

    legs = [[1., 1., 1., 1.], [0., 1., 1., 1.], [1., 0., 1., 1.]]

    for i in range(task_num):
        mass = i * 0.5 + 0.5
        SetBody([mass], "hopper_modified_" + str(i) + ".xml")

    envs = [[HopperEnv(xml_file = "hopper_modified_" + str(i) + ".xml") for _ in range(env_num)] for i in range(task_num)]
    state_len = envs[0][0].get_current_obs().size
    action_len = envs[0][0].action_space.shape[0]
    qpos_len = envs[0][0].model.nq
    traj_len = 10
    traj_track_len = 5
    latent_len = 3
    latent_body_len = 2
    latent_preserve = 4
    learner_batch = 32

    learner = Skill_Learner(state_len, action_len, traj_len, traj_track_len, latent_len, latent_body_len, qpos_len, task_num,
            learner_unity=args.learner_unity, learner_diverse=args.learner_diverse, sampler_diverse=args.sampler_diverse, sampler_disperse=args.sampler_disperse)
    follower = [Skill_Follower(state_len,  action_len, goal_len, str(i)) for i in range(task_num)]


    LOG_DIR = args.log_dir + "/" + CreateLogPrefix()
    log_file = CreateLogFile(LOG_DIR + "log.txt")

    sess = tf.Session()
    learner_saver = tf.train.Saver(max_to_keep=0, var_list=learner.trainable_dict)
    follower_savers = [tf.train.Saver(max_to_keep=0, var_list=follower[i].trainable_dict) for i in range(task_num)]

    init = tf.global_variables_initializer()
    sess.run(init)
    log_file.write("Episode\t")
    for i in range(task_num):
        log_file.write("Step"+str(i)+"\tMove"+str(i)+"\tReward"+str(i))
    log_file.write(learner.log_caption())
    for i in range(task_num):
        log_file.write(follower[i].log_caption())

    history_learner = [[], [], [], [], []]
    history_learner_noaction = []
    history_policy = [[], [], [], [], []]
    learner_weight = np.array([1., 1., 1., 1., 1.])


    with sess.as_default():

        learner.network_initialize()
        for i in range(task_num):
            follower[i].network_initialize()
        #learner_saver.restore(sess, "logs/humanoid_log0/extend/2022-04-08_04-23-34_log1_learner_2000.ckpt")
        #follower_savers[0].restore(sess, "logs/humanoid_log0/extend/2022-04-08_04-23-34_log1_follower0_2000.ckpt")
        #follower_savers[1].restore(sess, "logs/humanoid_log0/extend/2022-04-08_04-23-34_log1_follower1_2000.ckpt")
        #follower_savers[2].restore(sess, "logs/humanoid_log0/extend/2022-04-08_04-23-34_log1_follower2_2000.ckpt")



        for episode in range(1, 10001):

            cur_step = [0.] * task_num
            cur_move = [0.] * task_num
            cur_reward = [0.] * task_num
        
            for task in range(task_num):
                batch = 0
                while batch < 2 or len(history_learner[task]) < 4 * learner_batch:
                    cur_batch_step = 0.
                    cur_batch_move = 0.
                    cur_batch_reward = 0.

                                        
                    
                    states = []
                    for i in range(env_num):
                        state = envs[task][i].reset()
                        states.append(state)
                    survive_vector = [True]  * env_num
                    states = np.array(states)

                    goals = np.random.uniform(0.5, 2.0, size=(env_num, 1))

                    
                    for step in range(horizon):
                        if step % traj_len == 0:
                            state_traj = [[] for _ in range(env_num)]
                            action_traj = [[] for _ in range(env_num)]
                            body_traj = [[] for _ in range(env_num)]
                            first_traj_pos = states[:, :2]
                        if step % traj_track_len == 0:
                            first_track_pos = states[:, :2]



                        #regulized_goal = [relative_goal[i] / np.sqrt(np.sum(relative_goal[i] ** 2)) for i in range(env_num)]
                        actions = follower[task].get_action_batch(states, goals, False)
                        if batch == 0:
                            for i in range(env_num):
                                dropout = random.randrange(64)
                                if dropout < action_len:
                                    actions[i][dropout] = np.tanh(actions[i][dropout] + np.random.normal(0., 0.5))
                        #actions = np.clip(actions, -1., 1.)

                        new_states = []
                        for i in range(env_num):
                            state, reward, done, info = envs[task][i].step(actions[i])
                            body_difference = state[:2] - states[i][:2]

                            state_traj[i].append(states[i])
                            action_traj[i].append(actions[i])
                            if step % traj_track_len == traj_track_len - 1:
                                body_traj[i].append( state[:2] - first_track_pos[i] )
                            
                            if info['x_velocity'] < goals[i][0] :
                                reward += info['x_velocity'] * 0.5
                            elif info['x_velocity'] < goals[i][0] * 3. :
                                reward += (goals[i][0] * 0.75 - info['x_velocity'] * 0.25) 
                            if done == False:
                                reward += 0.5


                            cur_batch_move += np.abs(info['x_velocity'] - goals[i][0])
                            cur_batch_reward += reward
                            if survive_vector[i]:
                                history_policy[task].append([states[i], state, actions[i], goals[i], [reward], [0.0 if done else 1.0]])
                                cur_batch_step += 1

                            if done:
                                survive_vector[i] = False

                            new_states.append(state)
                        states = np.array(new_states)

                        if step % traj_len == traj_len - 1:
                            for i in range(env_num):
                                skill_moved = np.sqrt(np.sum((states[i][:2] - first_traj_pos[i]) ** 2))
                                
                                if skill_moved > 0.1:
                                    if batch != 0:
                                        history_learner[task].append([state_traj[i], body_traj[i], action_traj[i]])
                                    else:
                                        history_learner_noaction.append([state_traj[i], body_traj[i], action_traj[i]])

                                if survive_vector[i] == False   :
                                    state = envs[task][i].reset()
                                    states[i] = state
                                    survive_vector[i] = True
                    batch += 1
                                
                    cur_batch_move /= cur_batch_step
                    cur_batch_reward /= cur_batch_step
                    cur_batch_step /= env_num 
                    print("Episode " + str(episode) + " Task " + str(task) +  " Step " + str(cur_batch_step )+  " Move " +  str(cur_batch_move )
                        +  " Reward " +  str(cur_batch_reward ) )

                    cur_move[task] += cur_batch_move
                    cur_step[task] += cur_batch_step
                    cur_reward[task] += cur_batch_reward

                cur_move[task] /= batch
                cur_step[task] /= batch
                cur_reward[task] /= batch

            for iter in range(32):
                dic = random.sample(range(len(history_learner_noaction)), len(history_learner_noaction) if len(history_learner_noaction) < 64 else 64)

                state_dic = [history_learner_noaction[x][0] for x in dic]
                body_dic = [history_learner_noaction[x][1] for x in dic]
                action_dic = [history_learner_noaction[x][2] for x in dic]

                learner.optimize_batch_noaction(state_dic, body_dic, action_dic)

            for iter in range(32):
                dic = [random.sample(range(len(history_learner[task])), learner_batch) for task in range(task_num)]

                state_dic = np.concatenate([[history_learner[task][x][0] for x in dic[task]] for task in range(task_num)])
                body_dic = np.concatenate([[history_learner[task][x][1] for x in dic[task]] for task in range(task_num)])
                action_dic = np.concatenate([[history_learner[task][x][2] for x in dic[task]] for task in range(task_num)])

                learner.optimize_batch(state_dic, body_dic, action_dic)

            for task in range(task_num):
                for iter in range(64):
                    dic = random.sample(range(len(history_policy[task])), 64)

                    state_dic = [history_policy[task][x][0] for x in dic]
                    nextstate_dic = [history_policy[task][x][1] for x in dic]
                    action_dic = [history_policy[task][x][2] for x in dic]
                    goal_dic = [history_policy[task][x][3] for x in dic]
                    reward_dic = [history_policy[task][x][4] for x in dic]
                    survive_dic = [history_policy[task][x][5] for x in dic]

                    follower[task].optimize_batch(state_dic, nextstate_dic, action_dic, goal_dic, reward_dic, survive_dic, episode)



            learner.log_print()
            for i in range(task_num):
                follower[i].log_print()

            log_file.write(str(episode) + "\t")
            for i in range(task_num):
                log_file.write(str(cur_step[i]) + "\t" + str(cur_move[i])  + "\t" + str(cur_reward[i]) + "\t")
            log_file.write(learner.current_log())
            for i in range(task_num):
                log_file.write(follower[i].current_log())
            log_file.write("\n")
            log_file.flush()

            learner.network_update()
            for i in range(task_num):
                follower[i].network_update()

            if episode % 100 == 0:
                learner_saver.save(sess, LOG_DIR + "log1_learner_" + str(episode) + ".ckpt")
                for i in range(task_num):
                    follower_savers[i].save(sess, LOG_DIR + "log1_follower" + str(i) + "_" + str(episode) + ".ckpt")

            history_learner_noaction = history_learner_noaction[(len(history_learner_noaction) // 32):]
            for task in range(task_num):
                history_learner[task] = history_learner[task][(len(history_learner[task]) // 32):]
                history_policy[task] = history_policy[task][(len(history_policy[task] ) // 32):]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir')
    parser.add_argument('--learner-unity', type=float, default=0.01)
    parser.add_argument('--learner-diverse', type=float, default=0.01)
    parser.add_argument('--sampler-diverse', type=float, default=1.0)
    parser.add_argument('--sampler-disperse', type=float, default=0.0)

    args = parser.parse_args()
    Main(args)