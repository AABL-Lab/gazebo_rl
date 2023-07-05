# generic RL environment class

class RLEnv():
    # setup generic class to be used as a base for all RL environments
    def __init__(self, env_name, env_type, env_params):
        # store params
        self.env_name = env_name
        self.env_type = env_type
        self.env_params = env_params

        # initialize environment
        self.reset()

    def reset(self):
        # reset environment and setup for new episode
        raise NotImplementedError
    
    def step(self, action):
        # take action in environment
        raise NotImplementedError
    
    def get_obs(self):
        # return current observation
        raise NotImplementedError
    
    def get_reward(self):
        # return current reward
        raise NotImplementedError