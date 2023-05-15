import random
import numpy as np
import ipdb

class ReplayMemory:
    def __init__(self, capacity, seed, data=None):
        random.seed(seed)
        self.buffer = []
        if data is not None:
            for i in range(data['state'].shape[0]):
                self.buffer.append((data['state'][i], data['action'][i], data['reward'][i], data['next_state'][i], data['mask'][i], data['jac_ssa'][i], data['grad_rsa'][i]))
        self.capacity = max(capacity, len(self.buffer))
        self.position = len(self.buffer)

    def push(self, state, action, reward, next_state, done, jac_ssa, grad_rsa):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, jac_ssa, grad_rsa)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, jac_ssa, grad_rsa = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_jac(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, jac_ssa, grad_rsa = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, jac_ssa, grad_rsa

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
