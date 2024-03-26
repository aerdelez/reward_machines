import sys
import re
import multiprocessing
import os
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, VecEnv
import stable_baselines3.common.logger as logger
from stable_baselines3.common.logger import Logger
from importlib import import_module

# Importing our environments and auxiliary functions
import envs
from envs.water.water_world import Ball, BallAgent
from reward_machines.rm_environment import RewardMachineWrapper
from cmd_util import make_vec_env, make_env, common_arg_parser

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


pybullet_envs = None


roboschool = None

_game_envs = defaultdict(set)
for _, env in gym.envs.registry.items():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def get_session(config=None):
    """Get default session or create one with a given config"""
    sess = tf.get_default_session()
    if sess is None:
        sess = make_session(config=config, make_default=True)
    return sess

def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    """Returns a session that will use <num_cpu> CPU's only"""
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    if config is None:
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
        config.gpu_options.allow_growth = True

    if make_default:
        return tf.InteractiveSession(config=config, graph=graph)
    else:
        return tf.Session(config=config, graph=graph)

def train(args, extra_args, main_logger=None):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args, main_logger)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(main_logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    # Adding RM-related parameters
    alg_kwargs['use_rs']   = args.use_rs
    alg_kwargs['use_crm']  = args.use_crm
    alg_kwargs['gamma']    = args.gamma

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs,
        main_logger=main_logger
    )

    return model, env


def build_env(args, main_logger=None):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if alg in ['deepq', 'qlearning', 'hrm', 'dhrm']:
        env = make_env(env_id, env_type, args, seed=seed, logger_dir=main_logger.get_dir())
    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, args, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for _, env in gym.envs.registry.items():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    library = 'rl_agents'
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join([library, alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['baselines', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dictionary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        return logger.configure(log_path)
    else:
        return logger.configure(**kwargs)


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        main_logger = configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        main_logger = configure_logger(args.log_path, format_strs=[])

    model, env = train(args, extra_args, main_logger)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        main_logger.log("Running trained model")
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew
            env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew[i]))
                    episode_rew[i] = 0

    env.close()

    return model, main_logger

if __name__ == '__main__':

    # Examples over the office world:
    #    cross-product baseline: 
    #        >>> python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 
    #    cross-product baseline with reward shaping: 
    #        >>> python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --use_rs
    #    CRM: 
    #        >>> python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --use_crm
    #    CRM with reward shaping: 
    #        >>> python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --use_crm --use_rs
    #    HRM: 
    #        >>> python3.6 run.py --alg=hrm --env=Office-v0 --num_timesteps=1e5 --gamma=0.9
    #    HRM with reward shaping: 
    #        >>> python3.6 run.py --alg=hrm --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --use_rs
    # NOTE: The complete list of experiments (that we reported in the paper) can be found on '../scripts' 

    import time
    t_init = time.time()
    _, main_logger = main(sys.argv)
    main_logger.log("Total time: " + str(time.time() - t_init) + " seconds")