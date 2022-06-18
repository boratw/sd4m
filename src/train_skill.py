import numpy as np
import cv2
import tensorflow as tf
import random
from gym.envs.mujoco.ant_goal_leg_no_reward import AntGoalLegEnv
from networks.skill2_learner9 import Skill2_Learner
from networks.skill2_follower2 import Skill2_Follower
from util import CreateLogPrefix

gamma = 0.95
horizon = 200
env_num = 16
task_num = 4

legs = [0, 1, 2, 6]

envs = [[AntGoalLegEnv(i, [1., 1., 1., 1.]) for _ in range(env_num)] for i in range(task_num)]
state_len = envs[0][0].get_current_obs().size
action_len = envs[0][0].action_space.shape[0]
qpos_len = envs[0][0].model.nq
traj_len = 20
traj_track_len = 5
latent_len = 3
latent_body_len = 2
latent_preserve = 4
learner_batch = 32

learner = Skill2_Learner(state_len, action_len, traj_len, traj_track_len, latent_len, latent_body_len, qpos_len, task_num,
    sampler_diverse=1., sampler_disperse=0., learner_lr=0.001, sampler_lr=0.001, learner_unity=1., learner_diverse=2., learner_action_magnify=0.1)
follower = [Skill2_Follower(state_len,  action_len, str(i)) for i in range(task_num)]


LOG_DIR = "data/skill2/train12_2_2/" + CreateLogPrefix()
log_file = open(LOG_DIR + "log1.txt", "wt")

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
    
history_learner = [[] for _ in range(task_num)]
history_learner_noaction = []
history_policy =  [[] for _ in range(task_num)]


with sess.as_default():
    learner.network_initialize()
    for i in range(task_num):
        follower[i].network_initialize()
    #learner_saver.restore(sess, "data/skill2/train12/2021-09-03_09-15-12_log1_learner_1300.ckpt")
    follower_savers[0].restore(sess, "data/skill2/train12/2021-09-05_00-45-41_log1_follower0_3000.ckpt")
    follower_savers[1].restore(sess, "data/skill2/train12/2021-09-05_00-45-41_log1_follower1_3000.ckpt")
    follower_savers[2].restore(sess, "data/skill2/train12/2021-09-05_00-45-41_log1_follower2_3000.ckpt")
    follower_savers[3].restore(sess, "data/skill2/train12_2/2021-09-05_20-26-38_log1_follower3_1900.ckpt")



    for episode in range(1, 100001):

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

                random_weight = np.random.normal(0., 0.1, (env_num, 8)) 
                latent = np.random.normal(size=(env_num, latent_body_len))

                
                for step in range(horizon):
                    if step % traj_len == 0:
                        if (step // traj_len) % latent_preserve == 0:
                            latent = np.random.normal(size=(env_num, latent_body_len))
                            goal = learner.get_goal_batch(latent, True)
                            goal_swapped = np.array(np.swapaxes(goal,0,1))
                        state_traj = [[] for _ in range(env_num)]
                        action_traj = [[] for _ in range(env_num)]
                        body_traj = [[] for _ in range(env_num)]
                        relative_goal = np.zeros((env_num, 2), np.float32)
                        first_traj_pos = states[:, :2]
                    if step % traj_track_len == 0:
                        first_track_pos = states[:, :2]
                        relative_goal += goal_swapped[(step % traj_len) // traj_track_len]


                    states_noxy = states[:, 2:]

                    regulized_goal = [relative_goal[i] / np.sqrt(np.sum(relative_goal[i] ** 2)) for i in range(env_num)]
                    actions = follower[task].get_action_batch(states_noxy, regulized_goal, False)
                    if batch == 0:
                        for i in range(env_num):
                            dropout = random.randrange(32)
                            if dropout < action_len:
                                actions[i][dropout] = np.tanh(actions[i][dropout] + np.random.normal(0., 0.5))
                    #actions = np.clip(actions, -1., 1.)

                    new_states = []
                    for i in range(env_num):
                        state, reward, done, _ = envs[task][i].step(actions[i])
                        body_difference = state[:2] - states[i][:2]

                        state_traj[i].append(np.concatenate([ states[i][:2] - first_traj_pos[i], states_noxy[i]]))
                        action_traj[i].append(actions[i] * envs[task][i].leg_action)
                        if step % traj_track_len == traj_track_len - 1:
                            body_traj[i].append( state[:2] - first_track_pos[i] )
                            
                        reward += np.dot(body_difference, regulized_goal[i]) * 20.
                        cur_batch_reward += reward
                        if survive_vector[i]:
                            history_policy[task].append([states_noxy[i], state[2:], actions[i], regulized_goal[i], [reward], [0.0 if done else 1.0]])
                            cur_batch_step += 1
                        relative_goal[i] -= body_difference

                        if done:
                            survive_vector[i] = False

                        new_states.append(state)
                    states = np.array(new_states)

                    if step % traj_len == traj_len - 1:
                        for i in range(env_num):
                            skill_moved = np.sqrt(np.sum((states[i][:2] - first_traj_pos[i]) ** 2))
                            if skill_moved > 1.:
                                if batch != 0:
                                    history_learner[task].append([state_traj[i], body_traj[i], action_traj[i]])
                                else:
                                    history_learner_noaction.append([state_traj[i], body_traj[i], action_traj[i]])

                            if survive_vector[i] == False   :
                                state = envs[task][i].reset()
                                states[i] = state
                                survive_vector[i] = True
                            cur_batch_move += skill_moved
                batch += 1
                            

                cur_batch_step /= env_num                
                cur_batch_move /= env_num
                cur_batch_reward /= env_num
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