"""Tests for ExperimentRunner TF checkpoint save/restore (no meta graph)."""

import os
import tempfile

import tensorflow as tf


def _load_experiment_runner():
    from examples.development.main import ExperimentRunner
    return ExperimentRunner


def test_save_tf_session_checkpoint_omits_meta_graph():
    """write_meta_graph=False must not create a .meta file (GraphDef cap fix)."""
    ExperimentRunner = _load_experiment_runner()
    runner = ExperimentRunner.__new__(ExperimentRunner)
    with tf.Session() as sess:
        v = tf.Variable(3.0, name='test_var')
        sess.run(tf.global_variables_initializer())
        runner._session = sess
        with tempfile.TemporaryDirectory() as tmp:
            runner._save_tf_session_checkpoint(tmp)
            names = os.listdir(tmp)
            assert not any(name.endswith('.meta') for name in names), names
            assert any(name.startswith('tf_checkpoint.') for name in names), names


def test_restore_tf_session_checkpoint_roundtrip():
    """Weights restore after graph rebuild without a .meta file."""
    ExperimentRunner = _load_experiment_runner()
    runner = ExperimentRunner.__new__(ExperimentRunner)
    with tf.Session() as sess:
        v = tf.Variable(0.0, name='roundtrip_var')
        sess.run(tf.global_variables_initializer())
        runner._session = sess
        with tempfile.TemporaryDirectory() as tmp:
            sess.run(v.assign(42.0))
            runner._save_tf_session_checkpoint(tmp)
            sess.run(v.assign(0.0))
            assert float(sess.run(v)) == 0.0
            runner._restore_tf_session_checkpoint(tmp)
            assert float(sess.run(v)) == 42.0


def test_latest_tf_checkpoint_finds_weights():
    ExperimentRunner = _load_experiment_runner()
    runner = ExperimentRunner.__new__(ExperimentRunner)
    with tf.Session() as sess:
        v = tf.Variable(1.0, name='latest_var')
        sess.run(tf.global_variables_initializer())
        runner._session = sess
        with tempfile.TemporaryDirectory() as tmp:
            runner._save_tf_session_checkpoint(tmp)
            ckpt = runner._latest_tf_checkpoint(tmp)
            assert ckpt is not None
            assert ckpt.startswith(tmp)
