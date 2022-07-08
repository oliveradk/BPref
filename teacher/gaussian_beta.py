import numpy as np
from diversipy.hycusampling import stratify_generalized, stratified_sampling
from gym.spaces.box import Box
from scipy.stats import multivariate_normal
import json

import utils
from teacher import Teacher, Teachers


class Gaussian:
    def __init__(self, center, width, scale):
        self.center = center
        self.width = width
        self.scale = scale
        d = center.shape[0]
        self.var = multivariate_normal(mean=center, cov=np.identity(d) * width)
    
    def __call__(self, obs):
        return self.scale * self.var.pdf(obs)
    
    def __str__(self) -> str:
        return f"Gaussian: center {self.center}, width {self.width}, scale {self.scale}"


class GaussianBetaTeacher(Teacher):

    def __init__(self,
        ds,
        da,
        beta_func,
        obs_mask,  
        gamma, 
        eps_mistake,
        eps_skip,
        eps_equal
    ):
        super().__init__(ds, da, gamma, eps_mistake, eps_skip, eps_equal)
        self.beta_func = beta_func
        self.obs_mask = obs_mask
    
    def get_beta(self, sa_t, info_t):
        s = sa_t[:, :, :self.ds]
        p_s = s[:, :, self.obs_mask]
        return self.beta_func(p_s).mean(axis=1) #NOTE does this brodcast correctly?
    

class GaussianBetaTeachers(Teachers):
    
    VALID_DIMS = np.r_[0:18, 36:]
    WIDTH_DIVISOR = 2
    
    def __init__(self,
        n_teachers,
        ds, 
        da, 
        gamma, 
        eps_mistake, 
        eps_skip,
        eps_equal,
        width_divisor=2,
        beta_scale=1
    ):
        self.n_teachers = n_teachers
        self.width_divisor = width_divisor
        self.beta_scale = beta_scale
        self.params = {
            'ds': ds, 
            'da': da, 
            'gamma': utils.extend_param(gamma, self.n_teachers), 
            'eps_mistake': utils.extend_param(eps_mistake, self.n_teachers),
            'eps_skip': utils.extend_param(eps_skip, self.n_teachers), 
            'eps_equal': utils.extend_param(eps_equal, self.n_teachers)}
        super().__init__(teachers=[]) # teachers constructed in set_env
    
    def set_env(self, env, log_dir=None):
        self.define_teachers(env.observation_space, log_dir=log_dir)
        super().set_env(env)
    
            
    
    def define_teachers(self, obs_space, duplicates=False, log_dir=None):
            #preprocess environment space (remove duplicate and zero dimensions, normalize?)
            box, obs_mask = self._process_obs_space(obs_space, duplicates=duplicates)
            vol = utils.box_vol(box)
            #partition envionment space into n strata (don't need actual enviornment, only observation_space)
            strata = stratify_generalized(
                self.n_teachers, box.shape[0],  
                cuboid=(box.low.tolist(), box.high.tolist())
            )
            #sample points from strata
            points = stratified_sampling(strata)
            #calculate strata width for each
            strata_widths = utils.strata_width(strata)
            
            scale = self.beta_scale * vol/self.n_teachers
            for i in range(self.n_teachers):
                beta_func = Gaussian(points[i], strata_widths[i]/self.width_divisor, scale)
                teacher = GaussianBetaTeacher(
                    ds=self.params['ds'],
                    da=self.params['da'],
                    beta_func=beta_func,
                    obs_mask=obs_mask,
                    gamma=self.params['gamma'][i],
                    eps_mistake=self.params['eps_mistake'][i],
                    eps_skip=self.params['eps_skip'][i],
                    eps_equal=self.params['eps_equal'][i])
                self.teachers.append(teacher)
            if log_dir:
                self.log_teachers(log_dir) 

    def log_teachers(self, log_dir):
        teacher_list = []
        for teacher in self.teachers:
            teacher_list.append({
                'center': utils.arr_to_list(teacher.beta_func.center), 
                'width': utils.arr_to_list(teacher.beta_func.width),
                'scale': float(teacher.beta_func.scale)})
        with open(f'{log_dir}/teacher_data.json', 'w') as f:
            f.write(json.dumps(teacher_list))

    
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

        return p_box, obs_mask

