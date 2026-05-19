import tempfile

import tensorflow as tf


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
        model_data = state['model_str']
        if isinstance(model_data, str):
            model_data = model_data.encode('utf-8')

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(model_data)
            fd.flush()
            fd.seek(0)

            loaded_model = tf.keras.models.load_model(
                fd.name, custom_objects={
                    self.__class__.__name__: self.__class__})

        self.__dict__.update(loaded_model.__dict__.copy())

    @classmethod
    def from_config(cls, *args, custom_objects=None, **kwargs):
        custom_objects = custom_objects or {}
        custom_objects[cls.__name__] = cls
        custom_objects['tf'] = tf
        return super(PicklableKerasModel, cls).from_config(
            *args, custom_objects=custom_objects, **kwargs)
