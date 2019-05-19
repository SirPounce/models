import numpy as np
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from tensorflow.keras.backend import transpose


class Shuffle(Layer):
    def __init__(self, input_shape, out_channels, n_groups):
        super(Shuffle, self).__init__()
        self.n_groups = n_groups


    def call(self, input_tensor):

        _, h, w, n = input_tensor.shape.as_list()
        x = tf.reshape(input_tensor, [-1, h, w, self.n_groups, n // self.n_groups])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        return tf.reshape(x, [-1, h, w, n])


class PointWiseGroupConv(Layer):
    def __init__(self, in_channels, out_channels, n_groups):
        super(PointWiseGroupConv, self).__init__()
        self.n_groups = n_groups
        self.in_channels_group = in_channels // n_groups
        self.out_channels_group = math.ceil(out_channels / n_groups)
        self.concatenate = Concatenate(axis=-1)
        
        self.convs = []
        for group in range(n_groups):
          self.convs.append(
              Conv2D(
                       self.out_channels_group,  # filters
                       1,                   # kernel size
                       padding="valid",                   # should maybe be valid
                       use_bias=False
              )
          )


    def build(self, input_shape):
        


    def call(self, input_tensor):
        conved = []
        for i, conv in enumerate(self.convs):
            x = Lambda(lambda x:
                       x[
                           :,
                           :,
                           :,
                           i*self.in_channels_group:(i+1)*self.in_channels_group
              ]
            )(input_tensor)
            conved.append(self.convs[i](x))
            
        return self.concatenate(conved)

class ShuffleNetUnit(Layer):
    def __init__(self, input_shape, in_channels, out_channels, n_groups, stride):
        super(ShuffleNetUnit, self).__init__()
        self.stride = stride

        bottleneck_channels = out_channels // 4
        self.pw_gconv1 = PointWiseGroupConv(in_channels, bottleneck_channels, n_groups)
        self.pw_gconv2 = PointWiseGroupConv(bottleneck_channels, out_channels, n_groups)
        self.pw_gconv3 = PointWiseGroupConv(bottleneck_channels, out_channels-in_channels, n_groups)
        self.shuffle = Shuffle(input_shape, out_channels, n_groups)

        self.dw_conv1 = DepthwiseConv2D(3, 1, padding='same', depth_multiplier=1)
        self.dw_conv2 = DepthwiseConv2D(3, 2, padding='same', depth_multiplier=1)
        

    def call(self, input_tensor):
        print('input_tensor.shape after begin of ShuffleNetUnit: ', input_tensor.shape)

        layers = [BatchNormalization(), ReLU(),
                self.shuffle,
                self.dw_conv1, BatchNormalization(),
                self.pw_gconv2, BatchNormalization()]
        x = self.pw_gconv1(input_tensor)
        if self.stride < 2:
            for layer in layers:
                x = layer(x)
            x = Add()([x, input_tensor])
            print('x.shape end of ShuffleNetUnit with stride 1: ', x.shape)
            return ReLU()(x)
        else:
            layers[3] = self.dw_conv2
            layers[5] = self.pw_gconv3

            for layer in layers:
                x = layer(x)

            y = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(input_tensor)

            x = Concatenate()([x, y])
            print('x.shape end of shuffleNetUnit with stride 2: ', x.shape)
            return ReLU()(x)


def _vals():
        repeats = [3, 7, 3]
        out_channels = [[],
                [24, 144, 288, 576],
                [24, 200, 400, 800],
                [24, 240, 480, 960],
                [24, 272, 544, 1088],
                [],
                [],
                [],
                [24,384, 768, 1536]
        ]
        return repeats, out_channels


class ShuffleNet(Model):
    def __init__(self, input_shape, nb_classes, nb_groups, include_top=True, weights=None):
        super(ShuffleNet, self).__init__()

        repeats, out_channels = _vals()
        
        self.conv1 = Conv2D(
                out_channels[nb_groups][0],
                kernel_size=(3, 3),
                strides=2,
                use_bias=False,
                padding='same',
                activation='relu'
        )

        self.shuffle_units = []

        for i, reps in enumerate(repeats):
            self.shuffle_units.append(ShuffleNetUnit(
                input_shape,
                out_channels[nb_groups][i],             #input channels
                out_channels[nb_groups][i + 1],         #output 
                nb_groups,
                2                                       # changed stride
            ))
            for rep in range(reps):

                self.shuffle_units.append(ShuffleNetUnit(
                    input_shape,
                    out_channels[nb_groups][i],     
                    out_channels[nb_groups][i + 1],
                    nb_groups,
                    1
                ))
        
    def call(self, input_tensor):
        print('x.shape before conv1: ', input_tensor.shape)
        x = self.conv1(input_tensor)
        print('x.shape after conv1: ', x.shape)
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)    #Should this have softmax activation?
        print('x.shape after maxpool: ', x.shape)
        k=0
        for layer in self.shuffle_units:
            #print(f'x.shape after {0} runs in loop: ', x.shape)
            x = layer(x)
        x = GlobalAvgPool2D()(x)
        print('x.shape after Global Average Pool',x.shape)

        x = Dense(units=10, activation='softmax')(x)
        print('x.shape end of ShuffleNet',x.shape)
        return x     
      
