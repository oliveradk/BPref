import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse

import utils


def event_to_df(event_path, f_name='event_df.pkl') -> pd.DataFrame:
    par_dir = os.path.dirname(os.path.dirname(event_path))
    df_path = os.path.join(par_dir, f_name)
    if os.path.exists(df_path):
        return pd.read_pickle(df_path)
    
    ea = EventAccumulator(event_path).Reload() # 390 seconds (6m 30s) x 50 ~ 5 hours 
    tags = ea.Tags()['scalars']
    out = {}

    for tag in tags:
        tag_values=[]
        wall_time=[]
        steps=[]

        for event in ea.Scalars(tag):
            tag_values.append(event.value)
            wall_time.append(event.wall_time)
            steps.append(event.step)

        df = pd.DataFrame(data=dict(zip(steps,np.array([tag_values,wall_time]).transpose())), columns=steps,index=['value','wall_time'])
        df = utils.remove_duplicates(df)
        out[tag] = df
    
    return pd.concat(out.values(), keys=out.keys())


def get_config_events(config_exp_path):
    events = []
    for root, dirs, files in os.walk(config_exp_path):
        if "tb" in dirs:
            tb_dir = os.path.join(root, "tb")
            event = os.listdir(tb_dir)[0]
            events.append(os.path.join(tb_dir, event))
    return events


def save_dfs(dfs, paths, f_name='event_df.pkl'):
    assert len(dfs) == len(paths)

    for i in range(len(dfs)):
        save_dir = os.path.dirname(os.path.dirname(paths[i]))
        dfs[i].to_pickle(os.path.join(save_dir, f_name))

def build_df(filter_funcs, events):
    if len(filter_funcs) == 0:
        if len(events) != 1:
            raise ValueError("No filters for multiple events")
        return event_to_df(events[0])
    
    # get values of filter
    filter_vals = utils.apply_map(filter_funcs[0], events)

    # dict with unique filters as keys, list of events that share filter value as value
    event_filter_pairs = list(zip(events, filter_vals))
    event_groups = {filter_val: [pair[0] for pair in event_filter_pairs if pair[1] == filter_val] for filter_val in set(filter_vals)}
    df_dict = {}

    for filter_val, group in event_groups.items():
        df_dict[filter_val] = build_df(filter_funcs[1:], group)
    
    return pd.concat(df_dict.values(), keys=df_dict.keys())

### filters ### 

def teacher_int_param_filter(x, teacher_param):
     return re.search(f"(?<={teacher_param}': )\d+", x).group(0)

def teacher_str_param_filter(x, teacher_param):
    return re.search(f"(?<={teacher_param}': )'\w+'", x).group(0).replace("'", "")

def n_teacher_filter(x):
    return str(int(n_experts_filter(x)) + int(n_generalists_filter(x)))

def n_experts_filter(x):
    return teacher_int_param_filter(x, 'n_experts')

def n_generalists_filter(x):
    return teacher_int_param_filter(x, 'n_generalists')


def teacher_selection_filter(x):
    return re.search("(?<=teacher_selection).*(?=_state)", x).group(0)

def query_sample_filter(x):
    return re.search("(?<=sample).*(?=_teacher_selection)", x).group(0)

def seed_filter(x):
    return re.search("(?<=seed)\d+", x).group(0)

def gaussian_beta_filter(x):
    return 'CartpoleXGaussianTeachers' in x

def all_exps(x):
    return True

def four_teachers(x):
    re_query = re.search(f"(?<=n_experts': )\d+", x)
    if re_query is None:
        return False
    return re_query.group(0) == "4"

def sample_1_filter(x):
    re_query = re.search("(?<=sample).*(?=_teacher_selection)", x)
    if re_query is None:
        return False
    else:
        return re_query.group(0) == "1"



def get_events_in_matching(root_dir, filter):
    events = []
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            if filter(d):
                events += get_config_events(os.path.join(root, d))
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--root_filter', type=str)
    parser.add_argument('--filter_funcs', nargs='+')
    parser.add_argument('--build_subdfs', action='store_true')
    parser.add_argument('--build_df', action='store_true')
    parser.add_argument('--procs', type=int, default=4)
    parser.add_argument('--full_df_name', type=str)
    args = parser.parse_args()
    
    root_dir = args.root_dir
    root_filter = globals()[args.root_filter]
    filter_funcs = [globals()[filter_func] for filter_func in args.filter_funcs]
    build_subdfs = args.build_subdfs
    build_fulldf = args.build_df
    procs = args.procs
    if build_fulldf:
        full_df_name = args.full_df_name

    events = get_events_in_matching(root_dir, root_filter)
    if build_subdfs:
        with Pool(procs) as p:
            dfs = p.map(event_to_df, events)
        save_dfs(dfs, events)
    if build_fulldf:
        df = build_df(filter_funcs, events)
        df.to_pickle(os.path.join(root_dir, f"{full_df_name}.pkl"))
    
    


