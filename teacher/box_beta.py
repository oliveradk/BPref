import numpy as np
from diversipy.hycusampling import stratify_generalized, stratified_sampling
from gym.spaces.box import Box
from scipy.stats import multivariate_normal
import json

import utils
from teacher import Teacher, Teachers


class BoxBetaTeacher(Teacher):

    def __init__(self,
        ds,
        da,
        box,
        beta_high,
        beta_low,
        obs_mask,  
        gamma, 
        eps_mistake,
        eps_skip,
        eps_equal
    ):
        super().__init__(ds, da, gamma, eps_mistake, eps_skip, eps_equal)
        self.box = box
        self.beta_high = beta_high
        self.beta_low = beta_low
        self.obs_mask = obs_mask
    
    def get_beta(self, sa, info):
        s = sa[:, :self.ds]
        p_s = s[:, self.obs_mask]
        p_s = p_s.astype(self.box.dtype)
        betas = np.array([self.beta_high if self.box.contains(p_s_i) else self.beta_low for p_s_i in p_s])
        return betas.mean()  

class BoxBetaTeachers(Teachers):
    
    VALID_DIMS = np.r_[0:18, 36:]
    
    def __init__(self,
        n_teachers,
        ds, 
        da, 
        gamma, 
        eps_mistake, 
        eps_skip,
        eps_equal,
        beta_high,
        beta_low,
        scale,
    ):
        self.n_teachers = n_teachers
        self.scale = scale
        self.beta_high = beta_high if not self.scale else beta_high * n_teachers
        self.beta_low = beta_low
        self.params = {
            'ds': ds, 
            'da': da, 
            'gamma': utils.extend_param(gamma, self.n_teachers), 
            'eps_mistake': utils.extend_param(eps_mistake, self.n_teachers),
            'eps_skip': utils.extend_param(eps_skip, self.n_teachers), 
            'eps_equal': utils.extend_param(eps_equal, self.n_teachers)}
        super().__init__(teachers=[])
    
    def set_env(self, env, log_dir=None):
        self.define_teachers(env.observation_space)
        super().set_env(env)

    def define_teachers(self, obs_space, duplicates=False):
        #preprocess environment space (remove duplicate and zero dimensions, normalize)
        box, obs_mask, _norm_box = self._process_obs_space(obs_space, duplicates=duplicates)
        #partition envionment space into n strata (don't need actual enviornment, only observation_space)
        strata = stratify_generalized(
            self.n_teachers, box.shape[0],  
            cuboid=(box.low.tolist(), box.high.tolist())
        )
        for i in range(self.n_teachers):
            box = Box(low=np.array(strata[i][0]), high=np.array(strata[i][1]))
            teacher = BoxBetaTeacher(
                ds=self.params['ds'],
                da=self.params['da'],
                box=box,
                beta_high=self.beta_high,
                beta_low=self.beta_low,
                obs_mask=obs_mask,
                gamma=self.params['gamma'][i],
                eps_mistake=self.params['eps_mistake'][i],
                eps_skip=self.params['eps_skip'][i],
                eps_equal=self.params['eps_equal'][i])
            self.teachers.append(teacher)

    def _process_obs_space(self, obs_space, duplicates):
        #remove duplicate dimensions
        widths = utils.box_widths(obs_space)
        
        #unbounded/zero dims mask
        bounded_non_zero = np.logical_and(widths != np.inf, widths != 0)
        
        if duplicates:
            obs_mask = bounded_non_zero
        else:
            #valid dims mask
            valid_dims = np.zeros(obs_space.shape).astype(bool)
            valid_dims[self.VALID_DIMS] = True

            #conjoin masks
            obs_mask = np.logical_and(bounded_non_zero, valid_dims)

        #create new (processed) box
        p_box = Box(low=obs_space.low[obs_mask], high=obs_space.high[obs_mask])

        norm_box = Box(np.zeros(p_box.shape[0]), np.ones(p_box.shape[0]))

        return p_box, obs_mask, norm_box