import tensorflow as tf
import numpy as np
import copy
from .utils import flat_layer_weights_dict


def get_lr_values(optimizer):
    """This function is used to get the learning rates of the optimizer."""
    return optimizer.learning_rate

class TFScaffoldHelper(object):
    """Helper to be used with SCAFFOLD components."""

    def __init__(self):
        self.cnt = 0
        self.c_global = None
        self.c_local = None
        self.c_delta_para = None
        self.global_keys = None
        self.clip_norm = 1.0

    def init(self, model):
        self.c_global = tf.keras.models.clone_model(model)
        self.c_local = tf.keras.models.clone_model(model)
        # Initialize correction term with zeros
        c_init_para = {
            v.name: np.zeros_like(v.numpy()) for v in model.variables
        }
        self.c_global.set_weights([c_init_para[k] for k in c_init_para])
        self.c_local.set_weights([c_init_para[k] for k in c_init_para])

        # Generate a list of the flattened layers
        layer_weights_dict = {layer.name: layer.get_weights() for layer in self.c_global.layers}
        flattened_layer_weights_dict = flat_layer_weights_dict(layer_weights_dict)
        self.global_keys = [key for key, _ in flattened_layer_weights_dict.items()]
        print("Global")
        print(self.global_keys)

    def get_params(self):
        self.cnt = 0
        c_global_para = self.c_global.variables
        c_local_para = self.c_local.variables
        return c_global_para, c_local_para
        
 
    def model_update(self, model, curr_lr, c_global_para, c_local_para):
        net_para = model.variables  # Access only trainable variables
        trainable_var_names = [var.name for var in model.trainable_variables]

        for var in c_global_para:
            if not tf.reduce_all(tf.math.is_finite(var)):
                print(f"NaN/Inf found in c_global_para for variable {var.name}")

        for var in c_local_para:
            if not tf.reduce_all(tf.math.is_finite(var)):
                print(f"NaN/Inf found in c_local_para for variable {var.name}")

        new_weights = [
            tf.subtract(a, tf.multiply(curr_lr, tf.subtract(b, c)))
            for a, b, c in zip(net_para, c_global_para, c_local_para)
        ]

        for i, var in enumerate(net_para):
            new_weight = new_weights[i]
            if not tf.reduce_all(tf.math.is_finite(new_weight)):
                print(f"NaN/Inf found in new_weight for variable {new_weight}")
            else:
                if var.name in trainable_var_names:
                    var.assign(new_weight)

        self.cnt += 1
   
    def terms_update(self, model, curr_lr, c_global_para, c_local_para, model_global):
        c_new_para = self.c_local.variables
        self.c_delta_para = c_new_para
        global_model_para = model_global.variables
        net_para = model.variables
        scaler = 1 / (self.cnt * curr_lr)
        c_new_para = [
            tf.add(tf.subtract(a, b), tf.multiply(scaler, tf.subtract(c, d)))
            for a, b, c, d in zip(c_new_para, c_global_para, global_model_para, net_para)
        ]

        if c_global_para is None or c_local_para is None:
            raise ValueError("c_global_para or c_local_para is None!")

        for var in c_new_para:
            if not tf.reduce_all(tf.math.is_finite(var)):
                print(f"NaN/Inf found in c_new_para for variable")
                
        if tf.less(tf.constant(0, tf.float32), self.clip_norm):
            flatten_weights_delta = tf.nest.flatten(c_new_para)
            clipped_flatten_weights_delta, _ = tf.clip_by_global_norm(
                  flatten_weights_delta, clip_norm)
            c_new_para = tf.nest.pack_sequence_as(c_new_para,
                                                       clipped_flatten_weights_delta)
        c_delta_para_value = [tf.subtract(a, b) for a, b in zip(c_new_para, c_local_para)]

        self.c_delta_para = {
            self.global_keys[i]: c_delta_para_value[i].numpy()
            for i, var in enumerate(net_para)
        }

        for var in c_delta_para_value:
            if not tf.reduce_all(tf.math.is_finite(var)):
                print(f"NaN/Inf found in c_delta_para_value for variable")

        #self.c_local.set_weights(c_new_para)
        for var, weight in zip(self.c_local.variables, c_new_para):
             var.assign(weight)

    def load_global_controls(self, weights):
        weights_values = [v for _, v in weights.items()]
        self.c_global.set_weights(weights_values)

    def get_delta_controls(self):
        if self.c_delta_para is None:
            raise ValueError("c_delta_para hasn't been computed yet!")
            
        return self.c_delta_para

class ScaffoldCallback(tf.keras.callbacks.Callback):
    def __init__(self, scaffold_helper):
        super(ScaffoldCallback, self).__init__()
        self.scaffold_helper = scaffold_helper

    def on_train_begin(self, logs=None):
        self.c_global_para, self.c_local_para = self.scaffold_helper.get_params()

    def on_epoch_end(self, epoch, logs=None):
        curr_lr = self.model.optimizer.learning_rate
        self.scaffold_helper.model_update(
            self.model, curr_lr, self.c_global_para, self.c_local_para
        )
        print(f"SCAFFOLD model updated at end of epoch {epoch + 1}")
