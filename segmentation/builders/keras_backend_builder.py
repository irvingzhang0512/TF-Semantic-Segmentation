from ..backends.xception_deeplab import Xception


def build_keras_backend(backend_type, input_shape, OS=16):
    if backend_type == 'xception_deeplab':
        return Xception(input_shape=input_shape, OS=OS)
