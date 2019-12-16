import tensorflow as tf


def get_learning_rate(learning_policy,
                      base_learning_rate,

                      # exponential_decay
                      learning_rate_decay_step,
                      learning_rate_decay_factor,

                      # polynomial_decay
                      training_number_of_steps,
                      learning_power,
                      end_learning_rate,

                      # piecewise_constant_decay
                      learning_rate_boundaries,
                      learning_rate_values,

                      # slow start
                      slow_start_step,
                      slow_start_learning_rate,
                      slow_start_burnin_type='none'):
    global_step = tf.train.get_or_create_global_step()
    adjusted_global_step = global_step

    if slow_start_burnin_type != 'none':
        adjusted_global_step -= slow_start_step

    if learning_policy == 'step':
        learning_rate = tf.train.exponential_decay(
            base_learning_rate,
            adjusted_global_step,
            learning_rate_decay_step,
            learning_rate_decay_factor,
            staircase=True)
    elif learning_policy == 'poly':
        learning_rate = tf.train.polynomial_decay(
            base_learning_rate,
            adjusted_global_step,
            training_number_of_steps,
            end_learning_rate=end_learning_rate,
            power=learning_power)
    elif learning_policy == 'piecewise':
        learning_rate = tf.train.piecewise_constant_decay(
            adjusted_global_step,
            learning_rate_boundaries,
            learning_rate_values,
        )
    else:
        return base_learning_rate

    adjusted_slow_start_learning_rate = slow_start_learning_rate
    if slow_start_burnin_type == 'linear':
        adjusted_slow_start_learning_rate = (
            slow_start_learning_rate +
            (base_learning_rate - slow_start_learning_rate) *
            tf.cast(global_step, tf.float32) / slow_start_step)
    elif slow_start_burnin_type != 'none':
        raise ValueError('Unknown burnin type.')

    return tf.where(global_step < slow_start_step,
                    adjusted_slow_start_learning_rate, learning_rate)


def get_keras_learning_rate_fn(learning_policy,
                               base_learning_rate,

                               # exponential_decay
                               learning_rate_decay_step,
                               learning_rate_decay_factor,

                               # polynomial_decay
                               training_number_of_steps,
                               learning_power,
                               end_learning_rate,

                               # piecewise_constant_decay
                               learning_rate_boundaries,
                               learning_rate_values,):
    if learning_policy == 'step':
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=base_learning_rate,
            decay_steps=learning_rate_decay_step,
            decay_rate=learning_rate_decay_factor,
            staircase=True,
        )
    elif learning_policy == 'poly':
        return tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=base_learning_rate,
            decay_steps=training_number_of_steps,
            end_learning_rate=end_learning_rate,
            power=learning_power,
        )
    elif learning_policy == 'piecewise':
        print(learning_rate_boundaries, learning_rate_values)
        print(learning_rate_boundaries, learning_rate_values)
        print(learning_rate_boundaries, learning_rate_values)
        print(learning_rate_boundaries, learning_rate_values)
        print(learning_rate_boundaries, learning_rate_values)
        print(learning_rate_boundaries, learning_rate_values)
        print(learning_rate_boundaries, learning_rate_values)
        print(learning_rate_boundaries, learning_rate_values)
        print(learning_rate_boundaries, learning_rate_values)
        print(learning_rate_boundaries, learning_rate_values)
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=learning_rate_boundaries,
            values=learning_rate_values,
        )
    else:
        return base_learning_rate


def get_optimizer(opt_type, lr, **kwargs):
    if opt_type == 'adam':
        return tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
        )
    elif opt_type == 'sgd':
        return tf.keras.optimizers.SGD(
            learning_rate=lr,
            momentum=0.0,
            nesterov=False,
        )
    elif opt_type == 'rmsprop':
        return tf.keras.optimizers.RMSprop(
            learning_rate=lr,
            rho=0.9,
            momentum=0.0,
            epsilon=1e-7,
            centered=False,
        )
    raise ValueError('unknown optimizer type %s' % opt_type)
