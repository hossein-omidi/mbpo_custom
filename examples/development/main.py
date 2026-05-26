import os
import copy
import glob
import pickle
import sys
import pdb

import numpy as np
import tensorflow as tf
from ray import tune

from softlearning.environments.utils import get_environment_from_params
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant

from softlearning.misc.utils import set_seed, initialize_tf_variables
from examples.instrument import run_example_local

import mbpo.static

class ExperimentRunner(tune.Trainable):
    def _setup(self, variant):
        set_seed(variant['run_params']['seed'])

        self._variant = variant

        gpu_options = tf.GPUOptions(allow_growth=True)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        tf.keras.backend.set_session(session)
        self._session = tf.keras.backend.get_session()

        self.train_generator = None
        self._built = False
        self._best_eval_return = -np.inf
        self._best_eval_checkpoint_dir = None
        self._epochs_since_best_eval = 0

        save_every_epochs = self._variant['run_params'].get('save_every_epochs')
        if save_every_epochs is None:
            save_every_epochs = (
                self._variant.get('algorithm_params', {})
                .get('kwargs', {})
                .get('save_every_epochs'))
        self._save_every_epochs = (
            int(save_every_epochs)
            if save_every_epochs is not None else
            int(self._variant['run_params'].get('checkpoint_frequency', 0))
        )

        self._monitor_metric = (
            self._variant['run_params'].get('monitor_metric') or
            self._variant.get('algorithm_params', {})
                .get('kwargs', {})
                .get('monitor_metric', 'evaluation/return-average'))

    def _stop(self):
        tf.reset_default_graph()
        tf.keras.backend.clear_session()

    def _build(self):
        variant = copy.deepcopy(self._variant)

        environment_params = variant['environment_params']
        training_environment = self.training_environment = (
            get_environment_from_params(environment_params['training']))
        evaluation_environment = self.evaluation_environment = (
            get_environment_from_params(environment_params['evaluation'])
            if 'evaluation' in environment_params
            else training_environment)

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(variant, training_environment))
        sampler = self.sampler = get_sampler_from_variant(variant)
        Qs = self.Qs = get_Q_function_from_variant(
            variant, training_environment)
        policy = self.policy = get_policy_from_variant(
            variant, training_environment, Qs)
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy('UniformPolicy', training_environment))

        #### get termination function
        domain = environment_params['training']['domain']
        static_fns = mbpo.static[domain.lower()]
        ####
 
        self.algorithm = get_algorithm_from_variant(
            variant=self._variant,
            training_environment=training_environment,
            evaluation_environment=evaluation_environment,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            pool=replay_pool,
            static_fns=static_fns,
            sampler=sampler,
            session=self._session)

        initialize_tf_variables(self._session, only_uninitialized=True)

        self._built = True

    def _train(self):
        if not self._built:
            self._build()

        if self.train_generator is None:
            self.train_generator = self.algorithm.train()

        try:
            diagnostics = next(self.train_generator)
        except StopIteration:
            return {'done': True}

        self._maybe_checkpoint(diagnostics)

        return diagnostics

    def _maybe_checkpoint(self, diagnostics):
        current_epoch = getattr(self.algorithm, '_epoch', 0)
        if self._save_every_epochs and (current_epoch + 1) % self._save_every_epochs == 0:
            latest_checkpoint_dir = os.path.join(os.getcwd(), 'latest_checkpoint')
            print('[ ExperimentRunner ] Saving latest checkpoint to: {}'.format(latest_checkpoint_dir))
            self._save(latest_checkpoint_dir)

        q_loss = diagnostics.get('Q_loss')
        q_loss_warning_threshold = getattr(self.algorithm, '_q_loss_warning_threshold', None)
        if (q_loss_warning_threshold is not None and q_loss is not None and
                q_loss > q_loss_warning_threshold):
            emergency_checkpoint_dir = os.path.join(os.getcwd(), 'emergency_checkpoint')
            print('[ ExperimentRunner ] Q_loss {:.6f} > warning threshold {:.6f}. Saving emergency checkpoint to: {}'.format(
                q_loss, q_loss_warning_threshold, emergency_checkpoint_dir))
            self._save(emergency_checkpoint_dir)

        monitor_value = diagnostics.get(self._monitor_metric)
        if monitor_value is not None and monitor_value > self._best_eval_return:
            self._best_eval_return = monitor_value
            self._epochs_since_best_eval = 0
            best_checkpoint_dir = os.path.join(os.getcwd(), 'best_eval_checkpoint')
            self._best_eval_checkpoint_dir = best_checkpoint_dir
            print('[ ExperimentRunner ] New best eval {} = {:.6f}. Saving best checkpoint to: {}'.format(
                self._monitor_metric, monitor_value, best_checkpoint_dir))
            self._save(best_checkpoint_dir)
        else:
            self._epochs_since_best_eval += 1

    def _pickle_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint.pkl')

    def _replay_pool_pickle_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'replay_pool.pkl')

    def _tf_checkpoint_prefix(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint')

    def _get_tf_saver(self):
        return tf.train.Saver()

    @property
    def picklables(self):
        return {
            'variant': self._variant,
            'training_environment': self.training_environment,
            'evaluation_environment': self.evaluation_environment,
            'sampler': self.sampler,
            'algorithm': self.algorithm,
            'Qs': self.Qs,
            'policy_weights': self.policy.get_weights(),
        }

    def _save(self, checkpoint_dir):
        """Implements the checkpoint logic.

        TODO(hartikainen): This implementation is currently very hacky. Things
        that need to be fixed:
          - Figure out how serialize/save tf.keras.Model subclassing. The
            current implementation just dumps the weights in a pickle, which
            is not optimal.
          - Try to unify all the saving and loading into easily
            extendable/maintainable interfaces. Currently we use
            `tf.train.Checkpoint` and `pickle.dump` in very unorganized way
            which makes things not so usable.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        pickle_path = self._pickle_path(checkpoint_dir)
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.picklables, f)

        policy_weights_path = os.path.join(checkpoint_dir, 'policy_weights.pkl')
        with open(policy_weights_path, 'wb') as f:
            pickle.dump(self.policy.get_weights(), f)

        if self._variant['run_params'].get('checkpoint_replay_pool', False):
            self._save_replay_pool(checkpoint_dir)

        if hasattr(self.algorithm, '_model') and hasattr(self.algorithm._model, 'save'):
            try:
                self.algorithm._model.save(checkpoint_dir, getattr(self.algorithm, '_total_timestep', 0))
            except Exception as e:
                print('[ ExperimentRunner ] Warning: failed to save algorithm model: {}'.format(e))

        saver = self._get_tf_saver()
        # Use a distinct prefix for tf saver files to avoid name collisions
        tf_save_path = os.path.join(checkpoint_dir, 'tf_checkpoint')
        saver.save(self._session, tf_save_path)

        return os.path.join(checkpoint_dir, '')

    def _save_replay_pool(self, checkpoint_dir):
        replay_pool_pickle_path = self._replay_pool_pickle_path(
            checkpoint_dir)
        self.replay_pool.save_latest_experience(replay_pool_pickle_path)

    def _restore_replay_pool(self, current_checkpoint_dir):
        experiment_root = os.path.dirname(current_checkpoint_dir)

        experience_paths = [
            self._replay_pool_pickle_path(checkpoint_dir)
            for checkpoint_dir in sorted(glob.iglob(
                os.path.join(experiment_root, 'checkpoint_*')))
        ]

        for experience_path in experience_paths:
            try:
                self.replay_pool.load_experience(experience_path)
            except Exception as exc:
                raise RuntimeError(
                    'Failed to restore replay pool from {}. '
                    'Older replay buffers may be incompatible with the current '
                    'remaining_steps metadata required for exact PV rollout filtering. '
                    'Retrain from scratch or disable replay-pool restore.'.format(
                        experience_path)) from exc

    def _restore(self, checkpoint_dir):
        assert isinstance(checkpoint_dir, str), checkpoint_dir

        checkpoint_dir = checkpoint_dir.rstrip('/')

        with self._session.as_default():
            pickle_path = self._pickle_path(checkpoint_dir)
            with open(pickle_path, 'rb') as f:
                picklable = pickle.load(f)

        training_environment = self.training_environment = picklable[
            'training_environment']
        evaluation_environment = self.evaluation_environment = picklable[
            'evaluation_environment']

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(self._variant, training_environment))

        if self._variant['run_params'].get('checkpoint_replay_pool', False):
            self._restore_replay_pool(checkpoint_dir)

        sampler = self.sampler = picklable['sampler']
        Qs = self.Qs = picklable['Qs']
        # policy = self.policy = picklable['policy']
        policy = self.policy = (
            get_policy_from_variant(self._variant, training_environment, Qs))
        self.policy.set_weights(picklable['policy_weights'])
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy('UniformPolicy', training_environment))

        domain = self._variant['environment_params']['training']['domain']
        static_fns = mbpo.static[domain.lower()]

        self.algorithm = get_algorithm_from_variant(
            variant=self._variant,
            training_environment=training_environment,
            evaluation_environment=evaluation_environment,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            pool=replay_pool,
            static_fns=static_fns,
            sampler=sampler,
            session=self._session)
        self.algorithm.__setstate__(picklable['algorithm'].__getstate__())

        saver = self._get_tf_saver()
        saver.restore(self._session, tf.train.latest_checkpoint(
            os.path.split(self._tf_checkpoint_prefix(checkpoint_dir))[0]))
        initialize_tf_variables(self._session, only_uninitialized=True)

        # TODO(hartikainen): target Qs should either be checkpointed or pickled.
        for Q, Q_target in zip(self.algorithm._Qs, self.algorithm._Q_targets):
            Q_target.set_weights(Q.get_weights())

        if hasattr(self.algorithm, '_model') and hasattr(self.algorithm._model, 'load'):
            self._restore_model(checkpoint_dir)

        self._built = True

    def _restore_model(self, checkpoint_dir):
        if hasattr(self.algorithm, '_model') and hasattr(self.algorithm._model, 'load'):
            try:
                self.algorithm._model.load(checkpoint_dir)
            except Exception as e:
                print('[ ExperimentRunner ] Warning: failed to restore algorithm model from {}: {}'.format(checkpoint_dir, e))


def main(argv=None):
    """Run ExperimentRunner locally on ray.

    To run this example on cloud (e.g. gce/ec2), use the setup scripts:
    'softlearning launch_example_{gce,ec2} examples.development <options>'.

    Run 'softlearning launch_example_{gce,ec2} --help' for further
    instructions.
    """
    # __package__ should be `examples.development`
    package_name = __package__ or 'examples.development'
    run_example_local(package_name, argv)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
