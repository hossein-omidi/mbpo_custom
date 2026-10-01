"""Functions for instrumenting and running softlearning experiments locally.

Cloud/cluster launcher functions (GCE, EC2, autoscaler) have been removed —
this repo targets local execution only. The ray.autoscaler.commands module
no longer exists in Ray >= 1.0 and would crash on import.
"""

import importlib
import multiprocessing
import os
from pprint import pformat

import numpy as np
import ray
from ray import tune

from softlearning.misc.utils import datetimestamp


def _normalize_trial_resources(resources, cpu, gpu, extra_cpu, extra_gpu):
    if resources is None:
        resources = {}
    if cpu is not None:
        resources['cpu'] = cpu
    if gpu is not None:
        resources['gpu'] = gpu
    if extra_cpu is not None:
        resources['extra_cpu'] = extra_cpu
    if extra_gpu is not None:
        resources['extra_gpu'] = extra_gpu
    return resources


def add_command_line_args_to_variant_spec(variant_spec, command_line_args):
    variant_spec['run_params'].update({
        'checkpoint_frequency': (
            command_line_args.checkpoint_frequency
            if command_line_args.checkpoint_frequency is not None
            else variant_spec['run_params'].get('checkpoint_frequency', 0)
        ),
        'checkpoint_at_end': (
            command_line_args.checkpoint_at_end
            if command_line_args.checkpoint_at_end is not None
            else variant_spec['run_params'].get('checkpoint_at_end', True)
        ),
    })
    variant_spec['restore'] = command_line_args.restore
    return variant_spec


def generate_experiment(trainable_class, variant_spec, command_line_args):
    log_dir = command_line_args.log_dir or variant_spec.get('log_dir')
    domain = (
        variant_spec.get('domain')
        or variant_spec.get('environment_params', {})
                       .get('training', {})
                       .get('domain')
    )
    local_dir = (
        os.path.join(log_dir, domain)
        if log_dir and domain else log_dir or os.getcwd()
    )
    resources_per_trial = _normalize_trial_resources(
        command_line_args.resources_per_trial,
        command_line_args.trial_cpus,
        command_line_args.trial_gpus,
        command_line_args.trial_extra_cpus,
        command_line_args.trial_extra_gpus,
    )
    experiment_id = variant_spec.get('exp_name') or domain
    variant_spec = add_command_line_args_to_variant_spec(
        variant_spec, command_line_args)

    if command_line_args.video_save_frequency is not None:
        assert 'algorithm_params' in variant_spec
        variant_spec['algorithm_params']['kwargs']['video_save_frequency'] = (
            command_line_args.video_save_frequency)

    def create_trial_name_creator(trial_name_template=None):
        if not trial_name_template:
            return None
        def trial_name_creator(trial):
            return trial_name_template.format(trial=trial)
        return tune.function(trial_name_creator)

    experiment = {
        'run': trainable_class,
        'resources_per_trial': resources_per_trial,
        'config': variant_spec,
        'local_dir': local_dir,
        'num_samples': command_line_args.num_samples,
        'upload_dir': command_line_args.upload_dir,
        'checkpoint_freq': variant_spec['run_params']['checkpoint_frequency'],
        'checkpoint_at_end': variant_spec['run_params']['checkpoint_at_end'],
        'trial_name_creator': create_trial_name_creator(
            command_line_args.trial_name_template),
        'restore': command_line_args.restore,
    }
    return experiment_id, experiment


def get_experiments_info(experiments):
    number_of_trials = {
        experiment_id: len(list(
            tune.suggest.variant_generator.generate_variants(
                experiment_spec['config'])
        )) * experiment_spec['num_samples']
        for experiment_id, experiment_spec in experiments.items()
    }
    return {
        'number_of_trials': number_of_trials,
        'total_number_of_trials': sum(number_of_trials.values()),
    }


def confirm_yes_no(prompt):
    yes = {'yes', 'ye', 'y'}
    no = {'no', 'n'}
    choice = input(prompt).lower()
    while True:
        if choice in yes:
            return True
        elif choice in no:
            exit(0)
        else:
            print("Please respond with 'yes' or 'no'.\n(yes/no)")
        choice = input().lower()


def run_example_dry(example_module_name, example_argv):
    """Print the variant spec and related information of an example."""
    example_module = importlib.import_module(example_module_name)
    example_args = example_module.get_parser().parse_args(example_argv)
    variant_spec = example_module.get_variant_spec(example_args)
    trainable_class = example_module.get_trainable_class(example_args)
    experiment_id, experiment = generate_experiment(
        trainable_class, variant_spec, example_args)
    experiments = {experiment_id: experiment}
    experiments_info = get_experiments_info(experiments)

    print('\nDry run.\n\nExperiment specs:\n{}\n\nTrials: {}'.format(
        pformat(experiments, indent=2),
        experiments_info['total_number_of_trials'],
    ))

    training_env_params = (
        variant_spec.get('environment_params', {}).get('training', {}))
    if training_env_params.get('domain') == 'PVTracking':
        from softlearning.environments.utils import get_environment_from_params
        training_env = get_environment_from_params(training_env_params)
        obs = training_env.reset()
        inner = training_env.unwrapped
        kwargs = training_env_params.get('kwargs', {})
        print(
            'PVTracking env check: tz={!r} start_time={!r} periods={} '
            '({} steps), obs_mode={!r}, obs_space={}, reset_shape={}'.format(
                getattr(inner.location, 'tz', kwargs.get('tz', 'UTC')),
                getattr(inner, 'start_time', kwargs.get('start_time')),
                getattr(inner, 'periods', kwargs.get('periods')),
                getattr(inner, 'num_action_steps', '?'),
                kwargs.get('observation_mode', 'legacy'),
                training_env.observation_space.shape,
                np.asarray(obs).shape,
            ))
        training_env.close()


def run_example_local(example_module_name, example_argv, local_mode=False):
    """Run experiment locally; optionally parallelizes across cpus/gpus."""
    example_module = importlib.import_module(example_module_name)
    example_args = example_module.get_parser().parse_args(example_argv)
    variant_spec = example_module.get_variant_spec(example_args)
    trainable_class = example_module.get_trainable_class(example_args)
    experiment_id, experiment = generate_experiment(
        trainable_class, variant_spec, example_args)
    experiments = {experiment_id: experiment}

    ray.init(
        num_cpus=example_args.cpus,
        num_gpus=example_args.gpus,
        resources=example_args.resources or {},
        local_mode=local_mode,
        include_webui=example_args.include_webui,
        temp_dir=example_args.temp_dir,
        # Memory limits for 8GB node: cap object store + redis to leave headroom
        object_store_memory=500 * 1024 * 1024,   # 500 MB
        redis_max_memory=200 * 1024 * 1024,      # 200 MB
    )
    tune.run_experiments(
        experiments,
        with_server=example_args.with_server,
        server_port=4321,
        scheduler=None,
    )


def run_example_debug(example_module_name, example_argv):
    """Debug mode: run single-trial, non-parallelized (enables debugger use)."""
    debug_example_argv = []
    for option in example_argv:
        if '--trial-cpus' in option:
            available_cpus = multiprocessing.cpu_count()
            debug_example_argv.append('--trial-cpus={}'.format(available_cpus))
        elif '--upload-dir' in option:
            print('Ignoring {} in debug mode.'.format(option))
        else:
            debug_example_argv.append(option)
    run_example_local(example_module_name, debug_example_argv, local_mode=True)
