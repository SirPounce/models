from tensorflow.keras.layers import *
import tensorflow as tf




def ShuffleNetUnit(tensor, in_channels, out_channels, n_groups, stride): # [112,112,24], in_channels=24, n_groups=384, stride=2
  
  bottleneck_channels = out_channels // 4 #96
  x = pwGConv(tensor, in_channels, bottleneck_channels, n_groups)
  x = BatchNormalization()(x)
 
  x = Activation('relu')(x)
  
  x = Shuffle(x, n_groups) ####call this layer 
  
  
  print('after shuffle', type(x), x.shape)
  return x


def pwGConv(tensor, in_channels, out_channels, n_groups): # [112,112,24], in_channels=24, n_groups=384, stride=2
  #pointwise group convolution
  in_channels_group = in_channels // n_groups
  convs = []
  print('start of pwGC', type(tensor))

  
  for group in range(n_groups):
    x = Lambda(lambda x: x[:,
                           :,
                           :,
                            group*in_channels_group:(group+1)*in_channels_group])(tensor)

    out_channels_group = math.ceil(out_channels / n_groups)

    convs.append(
      Conv2D(filters=out_channels_group,
             kernel_size=[1,1],
             strides=(1, 1),
             padding="valid",
             use_bias=False)(x))
    
  out_tensor = Concatenate(axis=-1)(convs)
  
  print('after pwGC', type(out_tensor))
  return out_tensor
      
def Shuffle(tensor, n_groups):
  #shuffle channels
  _, h, w, n = tensor.shape
  out_channels_group = n // n_groups
  
  print('begin shuffle', type(tensor))
  
  x_reshaped = Reshape((h, w, n_groups, out_channels_group))(tensor)
  
  #print('x_reshaped')
  print('reshaped1 in shuffle', type(x_reshaped))
 
  
  x_transposed = Permute((1, 2, 4, 3))(x_reshaped)
  
  #print('x_transposed')
  print('transposed in shuffle', type(x_transposed))
  
  out_tensor = Reshape((h, w, n))(x_transposed)
  
  #print('out_tensor')
  print('end shuffle', type(out_tensor))
  
  return out_tensor

def DepthwiseConv(tensor, bottleneck_channels, stride):
  # Depthwise Separable Convolution
  print(tensor.shape)
  print('something is wrong')
  
  #---TODO: figure out if SeparableConv2D or DepthwiseConv2D is correct here????
  out_tensor = SeparableConv2D(filters=bottleneck_channels, kernel_size=[3,3], strides=(stride, stride), padding='same')(tensor)
  #out_tensor = DepthwiseConv2D(kernel_size=[3,3], strides=(stride, stride), padding='same', depth_multiplier=1, data_format='channels_last')(tensor)
  
  print('DW')
  return out_tensor


##############################
'''SHUFFLE NET ARCHITECTURE'''
##############################

def ShuffleNet(input_shape, n_classes, n_groups):
  settings = [[],
              [24, 144, 288, 576],
              [24, 200, 400, 800],
              [24, 240, 480, 960],
              [24, 272, 544, 1088],
              [],
              [],
              [],
              [24,384, 768, 1536]]
  
  inputs = Input(shape=(input_shape))
  print('inputs', type(inputs))
  
  '''2D Conv and MaxPool'''
  x = Conv2D( filters=24,
              kernel_size=[3,3],
              strides=(1,1),
              padding='same'
            )(inputs)
  
  print('x after conv2d', type(x))

  print(x.shape)
  x = MaxPooling2D(pool_size=[3, 3],
                    strides=1,
                    padding='same')(x)
  
  print('x after maxpool', type(x))

  
  '''Stages'''
  repetitions = [3, 7, 3]
  stages = settings[n_groups]
  for i, reps in enumerate(repetitions): 
      x = ShuffleNetUnit(x, stages[i], stages[i+1], n_groups, stride=2)
      for rep in range(reps):
        x = ShuffleNetUnit(x, stages[i], stages[i+1], n_groups, stride=1)
  
  print('after stages', x)

  '''Global Average Pooling'''

  x = GlobalAveragePooling2D()(x)
 
  print('after global average pooling', type(x))
  
  '''Dense Layers'''
  x = Dense(1000, activation='relu')(x)
  predictions = Dense(10, activation='softmax')(x)
              
  model = Model(inputs=inputs, outputs=predictions)
  
  return model
