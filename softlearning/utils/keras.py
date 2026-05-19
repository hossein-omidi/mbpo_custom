import os
import tempfile

import h5py
import tensorflow as tf


def _hdf5_text(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def _apply_keras_hdf5_compat_patches():
    """Patch TF 1.x HDF5 loaders when h5py returns str instead of bytes."""
    from tensorflow.python.keras.engine import saving

    if getattr(saving, '_mbpo_hdf5_compat', False):
        return

    def load_attributes_from_hdf5_group(group, name):
        if name in group.attrs:
            raw = group.attrs[name]
            try:
                return [_hdf5_text(item) for item in raw]
            except TypeError:
                return [_hdf5_text(raw)]

        data = []
        chunk_id = 0
        while '%s%d' % (name, chunk_id) in group.attrs:
            raw = group.attrs['%s%d' % (name, chunk_id)]
            if isinstance(raw, (list, tuple)):
                data.extend(_hdf5_text(item) for item in raw)
            else:
                data.append(_hdf5_text(raw))
            chunk_id += 1
        return data

    def load_weights_from_hdf5_group(f, layers):
        import numpy as np
        from tensorflow.python.keras import backend as K
        from tensorflow.python.keras.engine.saving import (
            preprocess_weights_for_loading,
        )

        if 'keras_version' in f.attrs:
            original_keras_version = _hdf5_text(f.attrs['keras_version'])
        else:
            original_keras_version = '1'
        if 'backend' in f.attrs:
            original_backend = _hdf5_text(f.attrs['backend'])
        else:
            original_backend = None

        filtered_layers = [layer for layer in layers if layer.weights]

        layer_names = load_attributes_from_hdf5_group(f, 'layer_names')
        filtered_layer_names = []
        for name in layer_names:
            g = f[name]
            weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
            if weight_names:
                filtered_layer_names.append(name)
        layer_names = filtered_layer_names
        if len(layer_names) != len(filtered_layers):
            raise ValueError(
                'You are trying to load a weight file containing %d layers '
                'into a model with %d layers.' % (
                    len(layer_names), len(filtered_layers)))

        weight_value_tuples = []
        for k, name in enumerate(layer_names):
            g = f[name]
            weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
            weight_values = [
                np.asarray(g[weight_name]) for weight_name in weight_names]
            layer = filtered_layers[k]
            symbolic_weights = layer.weights
            weight_values = preprocess_weights_for_loading(
                layer, weight_values, original_keras_version, original_backend)
            if len(weight_values) != len(symbolic_weights):
                raise ValueError(
                    'Weight count mismatch for layer %s.' % layer.name)
            weight_value_tuples += list(zip(symbolic_weights, weight_values))
        K.batch_set_value(weight_value_tuples)

    def load_model(filepath_or_model, custom_objects=None, compile=True):
        import json

        opened_new_file = not isinstance(filepath_or_model, h5py.File)
        if opened_new_file:
            h5_file = h5py.File(filepath_or_model, mode='r')
        else:
            h5_file = filepath_or_model

        try:
            model_config = h5_file.attrs.get('model_config')
            if model_config is None:
                raise ValueError('No model found in config file.')
            model_config = json.loads(_hdf5_text(model_config))
            model = saving.model_from_config(
                model_config, custom_objects=custom_objects)
            load_weights_from_hdf5_group(
                h5_file['model_weights'], model.layers)

            if compile:
                training_config = h5_file.attrs.get('training_config')
                if training_config is None:
                    return model
                training_config = json.loads(_hdf5_text(training_config))
                from tensorflow.python.keras import optimizers
                optimizer_config = training_config['optimizer_config']
                optimizer = optimizers.deserialize(
                    optimizer_config, custom_objects=custom_objects)
                loss = training_config['loss']
                metrics = training_config.get('metrics')
                loss_weights = training_config.get('loss_weights')
                sample_weight_mode = training_config.get('sample_weight_mode')
                model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=metrics,
                    loss_weights=loss_weights,
                    sample_weight_mode=sample_weight_mode)
            return model
        finally:
            if opened_new_file:
                h5_file.close()

    saving.load_attributes_from_hdf5_group = load_attributes_from_hdf5_group
    saving.load_weights_from_hdf5_group = load_weights_from_hdf5_group
    saving.load_model = load_model
    tf.keras.models.load_model = load_model
    saving._mbpo_hdf5_compat = True


class PicklableKerasModel(tf.keras.Model):
    def __getstate__(self):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            tf.keras.models.save_model(self, fd.name, overwrite=True)
            fd.flush()
            fd.seek(0)
            model_str = fd.read()
        d = {'model_str': model_str}

        return d

    def __setstate__(self, state):
        _apply_keras_hdf5_compat_patches()
        model_data = state['model_str']
        if isinstance(model_data, str):
            model_data = model_data.encode('utf-8')

        fd = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False)
        try:
            fd.write(model_data)
            fd.flush()
            fd.close()

            loaded_model = tf.keras.models.load_model(
                fd.name, custom_objects={
                    self.__class__.__name__: self.__class__})
        finally:
            os.unlink(fd.name)

        self.__dict__.update(loaded_model.__dict__.copy())

    @classmethod
    def from_config(cls, *args, custom_objects=None, **kwargs):
        custom_objects = custom_objects or {}
        custom_objects[cls.__name__] = cls
        custom_objects['tf'] = tf
        return super(PicklableKerasModel, cls).from_config(
            *args, custom_objects=custom_objects, **kwargs)
