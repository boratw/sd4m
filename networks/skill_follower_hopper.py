
import numpy as np
import tensorflow as tf
import math
from networks.gaussian_distribution import GaussianDistribution
from networks.gaussian_encoder import GaussianEncoder
from networks.gaussian_policy import GaussianPolicy
from networks.mlp import MLP


EPS = 1e-5

class Skill_Follower:
    def __init__(self, state_len, action_len, goal_len, name="",
        value_hidden_len=[1024, 1024], value_hidden_nonlinearity=tf.nn.leaky_relu, policy_hidden_len=[1024, 1024], policy_hidden_nonlinearity=tf.nn.tanh,
        value_lr=0.0001, policy_lr=0.0001, alpha_lr = 0.0001, policy_gamma=0.98, policy_reg=0.001, policy_update_ratio=0.05, learning_rate_decay=None) :

        self.name = "Skill_Follower" + name
        self.target_entropy=-action_len
        with tf.variable_scope(self.name): 
            self.input_state_noxy = tf.placeholder(tf.float32, [None, state_len], name="input_state_noxy")
            self.input_next_state_noxy = tf.placeholder(tf.float32, [None, state_len], name="input_next_state_noxy")
            self.input_goal = tf.placeholder(tf.float32, [None, goal_len], name="input_goal")
            self.input_action = tf.placeholder(tf.float32, [None, action_len], name="input_action")
            self.input_reward = tf.placeholder(tf.float32, [None, 1], name="input_reward")
            self.input_survive = tf.placeholder(tf.float32, [None, 1], name="input_survive")
            self.input_iter = tf.placeholder(tf.int32, [], name="input_iter")

            if learning_rate_decay is not None:
                value_lr = tf.train.exponential_decay(value_lr, self.input_iter, 100, learning_rate_decay)
                policy_lr = tf.train.exponential_decay(policy_lr, self.input_iter, 100, learning_rate_decay)
                alpha_lr = tf.train.exponential_decay(alpha_lr, self.input_iter, 100, learning_rate_decay)
                
            
            with tf.variable_scope("Follower"):
                stategoal = tf.concat([self.input_state_noxy, self.input_goal], axis=1)
                next_stategoal = tf.concat([self.input_next_state_noxy, self.input_goal], axis=1)

                self.log_alpha = tf.Variable(0., trainable=True)
                self.alpha = tf.exp(self.log_alpha)
                self.follower_policy = GaussianPolicy("follower_policy", state_len + goal_len, action_len, policy_hidden_len, hidden_nonlinearity=policy_hidden_nonlinearity,
                    input_tensor=stategoal, output_tanh=True)
                self.follower_next_policy = GaussianPolicy("follower_policy", state_len + goal_len, action_len, policy_hidden_len, hidden_nonlinearity=policy_hidden_nonlinearity,
                    input_tensor=next_stategoal, output_tanh=True, reuse=True)

                self.follower_qvalue1 = MLP("follower_qvalue1", state_len + goal_len, 1, value_hidden_len, hidden_nonlinearity=value_hidden_nonlinearity,
                    input_tensor=stategoal, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action)
                self.follower_qvalue2 = MLP("follower_qvalue2", state_len + goal_len, 1, value_hidden_len, hidden_nonlinearity=value_hidden_nonlinearity,
                    input_tensor=stategoal, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action)
                self.follower_qvalue1_policy = MLP("follower_qvalue1", state_len + goal_len, 1, value_hidden_len, hidden_nonlinearity=value_hidden_nonlinearity,
                    input_tensor=stategoal, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.follower_policy.reparameterized, reuse=True)
                self.follower_qvalue2_policy = MLP("follower_qvalue2", state_len + goal_len, 1, value_hidden_len, hidden_nonlinearity=value_hidden_nonlinearity,
                    input_tensor=stategoal, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.follower_policy.reparameterized, reuse=True)

                self.follower_qvalue1_target = MLP("follower_qvalue1_target", state_len + goal_len, 1, value_hidden_len, hidden_nonlinearity=value_hidden_nonlinearity,
                    input_tensor=next_stategoal, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.follower_next_policy.reparameterized)
                self.follower_qvalue2_target = MLP("follower_qvalue2_target", state_len + goal_len, 1, value_hidden_len, hidden_nonlinearity=value_hidden_nonlinearity,
                    input_tensor=next_stategoal, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.follower_next_policy.reparameterized)

                self.qvalue1_assign = self.follower_qvalue1_target.build_add_weighted(self.follower_qvalue1, 1.0)
                self.qvalue1_update = self.follower_qvalue1_target.build_add_weighted(self.follower_qvalue1, policy_update_ratio)
                self.qvalue2_assign = self.follower_qvalue2_target.build_add_weighted(self.follower_qvalue2, 1.0)
                self.qvalue2_update = self.follower_qvalue2_target.build_add_weighted(self.follower_qvalue2, policy_update_ratio)
                
                min_next_Q =  tf.minimum(self.follower_qvalue1_target.layer_output, self.follower_qvalue2_target.layer_output)
                Q_target = tf.stop_gradient(self.input_reward + (min_next_Q - self.follower_next_policy.log_pi * self.alpha)  * self.input_survive * policy_gamma)

                self.follower_qvalue1_optimizer = tf.train.AdamOptimizer(value_lr)
                self.follower_qvalue1_loss = tf.reduce_mean((self.follower_qvalue1.layer_output - Q_target) ** 2)
                self.follower_qvalue1_train = self.follower_qvalue1_optimizer.minimize(self.follower_qvalue1_loss,
                    var_list=self.follower_qvalue1.trainable_params)
                self.follower_qvalue2_optimizer = tf.train.AdamOptimizer(value_lr)
                self.follower_qvalue2_loss = tf.reduce_mean((self.follower_qvalue2.layer_output - Q_target) ** 2)
                self.follower_qvalue2_train = self.follower_qvalue2_optimizer.minimize(self.follower_qvalue2_loss,
                    var_list=self.follower_qvalue2.trainable_params)

                mean_Q = tf.reduce_mean([self.follower_qvalue1_policy.layer_output, self.follower_qvalue2_policy.layer_output], axis=0)
                self.follower_policy_loss = tf.reduce_mean(self.follower_policy.log_pi * self.alpha - mean_Q)  + self.follower_policy.regularization_loss * policy_reg
                self.follower_policy_optimizer = tf.train.AdamOptimizer(policy_lr)
                self.follower_policy_train = self.follower_policy_optimizer.minimize(self.follower_policy_loss,
                    var_list=self.follower_policy.trainable_params)


                self.follower_alpha_loss = tf.reduce_mean(-1. * (self.alpha * tf.stop_gradient(self.follower_policy.log_pi + self.target_entropy)))
                self.follower_alpha_optimizer = tf.train.AdamOptimizer(alpha_lr)
                self.follower_alpha_train = self.follower_alpha_optimizer.minimize(self.follower_alpha_loss,
                    var_list=[self.log_alpha])

                self.follower_qvalue1_average = tf.reduce_mean(self.follower_qvalue1.layer_output)
                self.follower_qvalue2_average = tf.reduce_mean(self.follower_qvalue2.layer_output)
                self.follower_policy_average = tf.reduce_mean(self.follower_policy.log_pi)
                    
                    


            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}

    def get_action(self, input_state_noxy, input_goal, discrete=False):
        input_list = {self.input_state_noxy : [input_state_noxy], self.input_goal : [input_goal]}
        sess = tf.get_default_session()
        if discrete:
            output = sess.run(self.follower_policy.mu, input_list)
        else:
            output = sess.run(self.follower_policy.reparameterized, input_list)
        return output[0]

    def get_action_batch(self, input_state_noxy, input_goal, discrete=False):
        input_list = {self.input_state_noxy : input_state_noxy, self.input_goal : input_goal}
        sess = tf.get_default_session()
        if discrete:
            output = sess.run(self.follower_policy.mu, input_list)
        else:
            output = sess.run(self.follower_policy.reparameterized, input_list)
        return output

    def optimize_batch(self, input_state_noxy, input_next_state_noxy, input_action, input_goal, input_reward, input_survive, input_iter=None):
        input_list = {self.input_state_noxy : input_state_noxy, self.input_next_state_noxy : input_next_state_noxy, 
            self.input_action : input_action, self.input_goal:input_goal, self.input_reward : input_reward, self.input_survive : input_survive,
            self.input_iter : input_iter}
        sess = tf.get_default_session()

        _, _, l1, l2 = sess.run([self.follower_qvalue1_train, self.follower_qvalue2_train,
            self.follower_qvalue1_average, self.follower_qvalue2_average], input_list)
        _, _, l3, l4 = sess.run([self.follower_policy_train, self.follower_alpha_train,
            self.follower_policy_average, self.alpha], input_list)

        self.log_policy_q1 += l1
        self.log_policy_q2 += l2
        self.log_policy_p += l3
        self.log_policy_a += l4
        self.log_num_follower += 1


    def network_initialize(self):
        sess = tf.get_default_session()
        sess.run([self.qvalue1_assign, self.qvalue2_assign])
        self.log_policy_q1 = 0
        self.log_policy_q2 = 0
        self.log_policy_p = 0
        self.log_policy_a = 0
        self.log_num_follower = 0

    def network_update(self):
        sess = tf.get_default_session()
        sess.run([self.qvalue1_update, self.qvalue2_update])
        self.log_policy_q1 = 0
        self.log_policy_q2 = 0
        self.log_policy_p = 0
        self.log_policy_a = 0
        self.log_num_follower = 0

    def log_caption(self):
        return "\t" + self.name + "_Avg_Qvalue1\t" + self.name + "_Avg_Qvalue2\t" + self.name + "_Avg_Policy\t"  + self.name + "_Avg_Alpha\t"
            
    
    def current_log(self):
        return "\t" + str(self.log_policy_q1 / self.log_num_follower) + "\t" + str(self.log_policy_q2 / self.log_num_follower) \
            + "\t" + str(self.log_policy_p / self.log_num_follower)  + "\t" + str(self.log_policy_a / self.log_num_follower) 

    def log_print(self):
        print ( self.name + "\n" \
            + "\tAvg_Qvalue1                      : " + str(self.log_policy_q1 / self.log_num_follower) + "\n" \
            + "\tAvg_Qvalue2                      : " + str(self.log_policy_q2 / self.log_num_follower) + "\n" \
            + "\tAvg_Policy                       : " + str(self.log_policy_p / self.log_num_follower) + "\n" \
            + "\tAvg_Alpha                        : " + str(self.log_policy_a / self.log_num_follower) )
