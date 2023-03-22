""" Various auxiliary utilities """
import math
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np
from models import MDRNNCell, VAE, Controller
import gym
import gym.envs.box2d

import cv2

from nvae.utils import add_sn
from nvae.vae_celeba import NVAE

# A bit dirty: manually change size of car racing env
gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 128, 128

# Hardcoded for now
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    3, 32, 256, 64, 64

# Same
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

transform_nvae = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def sample_continuous_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.

    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).

    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization

    :returns: sequence of seq_len actions
    """
    actions = [action_space.sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

class RolloutGenerator(object):
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, mdir, device, time_limit):
        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

        assert exists(vae_file) and exists(rnn_file),\
            "Either vae or mdrnn is untrained."

        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file)]

        for m, s in (('NVAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.nvae = NVAE(z_dim=256, img_dim=(128, 128))
        self.nvae.apply(add_sn)
        self.nvae.to(device)

        # print number of parameters in nvae
        print("Number of parameters in NVAE: {}".format(
            sum(p.numel() for p in self.nvae.parameters())))

        self.nvae.load_state_dict(torch.load(mdir + "/nvae/" + "ae_ckpt_28_0.902669.pth", map_location=device), strict=False)
        self.nvae.eval()

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})
        
        # print number of parameters in mdrnn
        print("Number of parameters in MDRNN: {}".format(
            sum(p.numel() for p in self.mdrnn.parameters())))

        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)

        # print number of parameters in controller
        print("Number of parameters in Controller: {}".format(
            sum(p.numel() for p in self.controller.parameters())))

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller...")
            self.controller.load_state_dict(ctrl_state['state_dict'])

        self.env = gym.make('CarRacing-v2')
        self.device = device

        self.time_limit = time_limit

        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])

        # create video writer from cv2
        self.writer = cv2.VideoWriter('output.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         50, (128, 128))
        
        self.dream_writer = cv2.VideoWriter('dream_output.avi',
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            50, (128, 128))

    def get_action_and_transition(self, obs, hidden, obs_nvae):
        """ Get action and transition.

        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        recon_obs, latent_mu, _ = self.vae(obs)

        recon_obs = recon_obs.squeeze().permute(1, 2, 0).cpu().numpy()

        # convert to uint8
        recon_obs = (recon_obs * 255).astype(np.uint8)
        # convert to BGR
        recon_obs = cv2.cvtColor(recon_obs, cv2.COLOR_RGB2BGR)
        # write to video
        # self.dream_writer.write(recon_img)

        action = self.controller(latent_mu, hidden[0])
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)
        return action.squeeze().cpu().numpy(), next_hidden

    def rollout(self, params, render=False):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        obs, _ = self.env.reset()

        # This first render is required !
        # self.env.render()

        hidden = [
            torch.zeros(1, RSIZE).to(self.device)
            for _ in range(2)]

        cumulative = 0
        i = 0

        dream_imgs = []

        while True:
            print(f"Rendering frame {i}      ",end='\r')
            self.writer.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            temp_obs = obs.copy()
            dream_imgs.append(temp_obs[:112, :, :])
            obs = transform(obs[:112, :, :]).unsqueeze(0).to(self.device)
            obs_nvae = transform_nvae(temp_obs[:112, :, :]).unsqueeze(0).to(self.device)
            action, hidden = self.get_action_and_transition(obs, hidden, obs_nvae)
            obs, reward, done, _, _ = self.env.step(action)

            if render:
                self.env.render()

            cumulative += reward
            if done or i > self.time_limit:
                self.writer.release()
                
                nimgs = np.array(dream_imgs)
                # convert to torch tensor
                nimgs = torch.from_numpy(nimgs).permute(0, 3, 1, 2).float() / 255
                # resize to 128x128
                nimgs = torch.nn.functional.interpolate(nimgs, size=(128, 128))
                print(nimgs.shape)
                gen_imgs, _, _, _, _ = self.nvae(nimgs.to(self.device))
                gen_imgs = gen_imgs.permute(0, 2, 3, 1)
                gen_imgs = gen_imgs.cpu().numpy() * 255
                gen_imgs = gen_imgs.astype(np.uint8)
                for i in range(gen_imgs.shape[0]):
                    self.dream_writer.write(cv2.cvtColor(gen_imgs[i], cv2.COLOR_RGB2BGR))
                self.dream_writer.release()

                return - cumulative
            i += 1