import numpy as np

class ContextualOptimalPolicy():
    def __init__(self, reward_gold, prior_prob = 0.5):
        # Samples initial policy from a uniform prior at random then learns about the tiger so take optimal policy after
        sample = np.random.rand()
        self.reward_gold = reward_gold
        if sample < 0.5:
            self.action = 'OPEN_RIGHT'
        else:
            self.action = 'OPEN_LEFT'

    def take_action(self):
        return self.action

    def update_posterior(self, state, reward):
        if reward != self.reward_gold:
            if self.action == 'OPEN_RIGHT':
                self.action = 'OPEN_LEFT'
            else:
                self.action = 'OPEN_RIGHT'

    def reset(self):
        sample = np.random.rand()
        if sample < 0.5:
            self.action = 'OPEN_RIGHT'
        else:
            self.action = 'OPEN_LEFT'






class BayesOptimalPolicy():
    def __init__(self, reward_gold, reward_tiger, reward_listen, state_accuracy, horizon):

        self.reward_gold = float(reward_gold)
        self.reward_tiger = float(reward_tiger)
        self.reward_listen = float(reward_listen)
        self.N_right = 0
        self.N_left = 0
        self.t = 0
        self.door_opened = False
        self.p = float(state_accuracy)
        self.T = horizon
        self.max_posterior_action = "OPEN_LEFT"


    def predictive_return(self, t):
        num = self.sup_posterior() * (self.p ** (self.p * (t))) * ((1-self.p) ** ((1-self.p) * (t) ))
        denom =  num + ((1-self.sup_posterior()) * ((1 - self.p) ** (self.p * (t))) * ((self.p) ** ((1-self.p) * (t))))
        prob_gold = num / denom
        prob_tiger = 1 - prob_gold
        predictive_reward = prob_gold * self.reward_gold + prob_tiger * self.reward_tiger
        return  predictive_reward + (self.T - t - 1) * self.reward_gold

    def update_posterior(self, state, reward):
        self.t += 1
        if state == "GROWL_LEFT":
            self.N_left += 1
            self.update_max_posterior_action()
        if state == "GROWL_RIGHT":
            self.N_right += 1
            self.update_max_posterior_action()
        if self.door_opened:
            if reward != self.reward_gold:
                if self.max_posterior_action == 'OPEN_RIGHT':
                    self.max_posterior_action = 'OPEN_LEFT'
                else:
                    self.max_posterior_action = 'OPEN_RIGHT'


    def update_max_posterior_action(self):
        if self.N_right > self.N_left:
            self.max_posterior_action = "OPEN_LEFT"
        else:
            self.max_posterior_action = "OPEN_RIGHT"

    def sup_posterior(self):
        evidence = (self.p **  self.N_left) * ((1-self.p) ** self.N_right) + (self.p **  self.N_right) * ((1-self.p) ** self.N_left)
        if self.N_left > self.N_right:
            num = (self.p **  self.N_left) * ((1-self.p) ** self.N_right)
        else:
            num = (self.p **  self.N_right) * ((1-self.p) ** self.N_left)
        return num / evidence

    def take_action(self):

        if self.door_opened:
            return self.max_posterior_action
        else:
            Qimax = 0.0
            for i in range(self.T-self.t-1):
                Qimax = max(self.predictive_return(self.T-1-i), self.reward_listen + Qimax)
            if self.predictive_return(self.t) > self.reward_listen + Qimax:
                self.door_opened = True
                return self.max_posterior_action
            else:
                return "LISTEN"

    def reset(self):
        self.N_right = 0
        self.N_left = 0
        self.t = 0
        self.door_opened = False

