
import numpy as np
import tensorflow as tf
import math
from datetime import datetime
import scipy.signal

from trpo.policy import Policy
from trpo.value_function import NNValueFunction
from trpo.utils import Scaler

EPS = 1e-5

class Explorer_TRPO:
    def __init__(self, state_len, action_len, goal_len, name="", env_num=1, policy_gamma=0.99, lam=0.98, kl_targ=0.003, hid1_mult=10,
        policy_logvar=-1.0, restore_mean=None, restore_var=None, min_trajectory=2) :


        self.name = "Explorer_TRPO" + name
        with tf.variable_scope(self.name): 
            self.val_func = NNValueFunction(state_len + goal_len + 1, hid1_mult)
            self.policy = Policy(state_len + goal_len + 1, action_len, kl_targ, hid1_mult, policy_logvar)
            #self.scaler = Scaler(state_len, restore_mean, restore_var)
            self.policy._build_graph()



            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}
        
        print(self.trainable_dict)
        self.trajectories = []
        self.tmp_observes = [[] for _ in range(env_num)]
        self.tmp_actions = [[] for _ in range(env_num)]
        self.tmp_rewards = [[] for _ in range(env_num)]
        self.gamma = policy_gamma
        self.lam = lam
        self.min_trajectory = min_trajectory
        self.env_num = env_num
        self.time_step = 0.

    def get_action(self, input_state, input_goal, record=True):
        time_step_array = np.array([ [self.time_step] for _ in range(self.env_num)])
        obs = np.concatenate([np.array([input_state]), np.array([input_goal]), time_step_array], axis=1)
        action = self.policy.sample(obs)
        if record:
            self.tmp_observes[0].append(obs)
            self.tmp_actions[0].append(action)
        self.time_step += 1.
        return action[0]

    def get_action_batch(self, input_state, input_goal, survive_vector=None, record=True):
        time_step_array = np.array([ [self.time_step] for _ in range(self.env_num)])
        obs = np.concatenate([np.array(input_state), np.array(input_goal), time_step_array], axis=1)
        action = self.policy.sample(obs)
        if record:
            for i in range(self.env_num):
                if survive_vector[i] == True:
                    self.tmp_observes[i].append(obs[i])
                    self.tmp_actions[i].append(action[i])
        self.time_step += 1.
        return action

    def push_reward(self, input_reward):
        self.tmp_rewards[0].append(np.array(input_reward))

    def push_reward_batch(self, input_reward, survive_vector):
        for i in range(self.env_num):
            if survive_vector[i] == True:
                self.tmp_rewards[i].append(np.array(input_reward[i]))

    def episode_finished(self):
        for i in range(self.env_num):
            if len(self.tmp_observes[i]) > self.min_trajectory:
                trajectory = {'observes': np.array(self.tmp_observes[i]),
                            'actions': np.array(self.tmp_actions[i]),
                            'rewards': np.array(self.tmp_rewards[i]),
                            'reward_sum': np.sum(self.tmp_rewards[i])}
                self.trajectories.append(trajectory)
        #self.scaler.update(trajectory['unscaled_obs'])

        self.tmp_observes = [[] for _ in range(self.env_num)]
        self.tmp_actions = [[] for _ in range(self.env_num)]
        self.tmp_rewards = [[] for _ in range(self.env_num)]
        self.time_step = 0.

    def optimize_batch(self, epsilon=1.0):

        if epsilon < 1.0:
            m = np.percentile([t['reward_sum'] for t in self.trajectories], epsilon * 100)
            self.trajectories = [t for t in self.trajectories if t['reward_sum'] < m ]

        add_value(self.trajectories, self.val_func)  # add estimated values to episodes
        add_disc_sum_rew(self.trajectories, self.gamma)  # calculated discounted sum of Rs
        add_gae(self.trajectories, self.gamma, self.lam)  # calculate advantage
        observes, actions, advantages, disc_sum_rew = build_train_set(self.trajectories)
        ret_policy = self.policy.update(observes, actions, advantages)  # update policy
        ret_value = self.val_func.fit(observes, disc_sum_rew)  # update value function
        
        self.log_policy_loss = ret_policy['PolicyLoss']
        self.log_policy_entropy = ret_policy['PolicyEntropy']
        self.log_policy_kl = ret_policy['KL']
        self.log_policy_beta = ret_policy['Beta']
        self.log_value_loss = ret_value['ValFuncLoss']
        self.log_value_newvar = ret_value['ExplainedVarNew']
        self.log_value_oldvar = ret_value['ExplainedVarOld']

        self.trajectories = []

    def network_initialize(self):
        with tf.variable_scope(self.name): 
            self.policy.sess =  tf.get_default_session()
        self.log_policy_loss = 0
        self.log_policy_entropy = 0
        self.log_policy_kl = 0
        self.log_policy_beta = 0
        self.log_value_loss = 0
        self.log_value_newvar = 0
        self.log_value_oldvar = 0

    def network_update(self):
        self.log_policy_loss = 0
        self.log_policy_entropy = 0
        self.log_policy_kl = 0
        self.log_policy_beta = 0
        self.log_value_loss = 0
        self.log_value_newvar = 0
        self.log_value_oldvar = 0

    def log_caption(self):
        return "\t" + self.name + "_PolicyLoss\t" + self.name + "_PolicyEntropy\t" + self.name + "_KL\t"  + self.name + "_Beta\t" \
            + self.name + "_ValFuncLoss\t" + self.name + "_ExplainedVarNew\t" + self.name + "_ExplainedVarOld\t"
            
    
    def current_log(self):
        return "\t" + str(self.log_policy_loss) + "\t" + str(self.log_policy_entropy) + "\t" + str(self.log_policy_kl)  + "\t" + str(self.log_policy_beta) \
            + "\t" + str(self.log_value_loss)  + "\t" + str(self.log_value_newvar)  + "\t" + str(self.log_value_oldvar) 

    def log_print(self):
        print ( self.name + "\n" \
            + "\tPolicyLoss               : " + str(self.log_policy_loss) + "\n" \
            + "\tPolicyEntropy            : " + str(self.log_policy_entropy) + "\n" \
            + "\tKL                       : " + str(self.log_policy_kl) + "\n" \
            + "\tBeta                     : " + str(self.log_policy_beta)  + "\n" \
            + "\tValFuncLoss              : " + str(self.log_value_loss)  + "\n" \
            + "\tExplainedVarNew          : " + str(self.log_value_newvar)  + "\n" \
            + "\tExplainedVarOld          : " + str(self.log_value_oldvar) )



def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories
    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value
    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values.reshape((-1))

def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories
    Args:
        trajectories: as returned by run_policy()
        gamma: discount
    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf
    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)
    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages

def build_train_set(trajectories):
    """
    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()
    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew