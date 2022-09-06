"""
Andrew Player
September 2022
Basic convolutional model, for now, for classifying the images.
"""

from tensorflow                  import Tensor
from tensorflow.keras.layers     import Conv2D, Input, LeakyReLU, Flatten, Dense
from tensorflow.keras.models     import Model
from tensorflow.keras.optimizers import SGD

def conv2d_block(
    input_tensor: Tensor ,
    num_filters:  int
) -> Tensor:

    """
    2D-Convolution Block for encoding / generating feature maps.
    """

    x = Conv2D(
        filters            = num_filters,
        kernel_size        = (3, 3)     ,
        kernel_initializer = 'he_normal',
        padding            = 'same'     ,
    )(input_tensor)

    x = LeakyReLU()(x)

    return x


def create_net(
    model_name:    str   = 'model',
    tile_size:     int   = 128    ,
    num_filters:   int   = 32     ,
    label_count:   int   = 7
) -> Model:

    """
    Creates a basic convolutional network
    """

    input = Input(shape = (tile_size, tile_size, 1))


    # # --------------------------------- #
    # # Feature Map Generation            #
    # # --------------------------------- #

    c1 = conv2d_block(input, num_filters * 1)
    c2 = conv2d_block(c1   , 1)

    # # --------------------------------- #
    # # Dense Hidden Layer                #
    # # --------------------------------- #

    f0 = Flatten()(c2)
    d0 = Dense(c2.shape[1] * c2.shape[2])(f0)
    d0 = LeakyReLU()(d0)

    # --------------------------------- #
    # Output Layer                      #
    # --------------------------------- #

    output = Dense(label_count)(d0)
    output = LeakyReLU()(output)

    # --------------------------------- #
    # Model Creation and Compilation    #
    # --------------------------------- #

    model = Model(
        inputs  = [input ],
        outputs = [output], 
        name    = model_name
    )

    model.compile(
        optimizer = SGD(learning_rate=0.005),
        loss      = 'mean_squared_error',
        metrics   = ['mean_squared_error'],
    )

    return model