
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow as tf
from Attention import AttentionModule
from config import *

class Bottleneck(layers.Layer):

    def __init__(self, output_dim, strides=1, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.conv1 = layers.Conv2D(output_dim // 4, kernel_size=1, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(output_dim // 4, kernel_size=3, strides=strides, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(output_dim, kernel_size=1, padding="same", use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def build(self, input_shape):
        super(Bottleneck, self).build(input_shape)

    def call(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = self.relu(out)

        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class RFRModule(layers.Layer):

    def __init__(self, layer_size=6, in_channel=64, **kwargs):
        super(RFRModule, self).__init__(**kwargs)
        self.freeze_enc_bn = False
        self.layer_size = layer_size
        self.encoder_blocks = {}
        self.in_channel = in_channel
        for _ in range(3):
            out_channel = in_channel * 2
            block = models.Sequential(
                [layers.Conv2D(out_channel, kernel_size=3, strides=2, padding="same", use_bias=False)
                    , layers.BatchNormalization(), layers.ReLU()])
            name = "enc_{:d}".format(_ + 1)
            self.encoder_blocks[name] = block
            in_channel = out_channel

        for _ in range(3, 6):
            block = models.Sequential(
                [layers.Conv2D(out_channel, kernel_size=3, strides=1, padding="same", dilation_rate=2, use_bias=False)
                    , layers.BatchNormalization(), layers.ReLU()])
            name = "enc_{:d}".format(_ + 1)
            self.encoder_blocks[name] = block
        # 知识一致注意力模块 TODO
        self.att = AttentionModule(512)

        self.decoder_blocks = {}
        for _ in range(5, 3, -1):
            block = models.Sequential(
                [layers.Conv2D(in_channel, kernel_size=3, strides=1, padding="same", dilation_rate=2, use_bias=False)
                    , layers.BatchNormalization(), layers.LeakyReLU(0.2)])
            name = "dec_{:d}".format(_)
            self.decoder_blocks[name] = block

        # 1024 -> 512
        self.decoder_blocks["dec_3"] = models.Sequential([
            layers.Conv2DTranspose(8 * self.in_channel, kernel_size=4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
        # 768 -> 256
        self.decoder_blocks["dec_2"] = models.Sequential([
            layers.Conv2DTranspose(4 * self.in_channel, kernel_size=4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
        # 384 ->64
        self.decoder_blocks["dec_1"] = models.Sequential([
            layers.Conv2DTranspose(self.in_channel, kernel_size=4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])

    def build(self, input_shape):
        super(RFRModule, self).build(input_shape)

    def call(self, input, mask):

        h_dict = {}
        h_dict["h_0"] = input

        h_key_prev = "h_0"
        for i in range(1, self.layer_size + 1):
            l_key = "enc_{:d}".format(i)
            h_key = "h_{:d}".format(i)
            h_dict[h_key] = self.encoder_blocks[l_key](h_dict[h_key_prev])
            h_key_prev = h_key

        h = h_dict[h_key]

        for i in range(self.layer_size - 1, 0, -1):
            enc_h_key = "h_{:d}".format(i)
            dec_l_key = "dec_{:d}".format(i)
            h = tf.concat([h, h_dict[enc_h_key]], axis=-1)
            h = self.decoder_blocks[dec_l_key](h)
            if (i == 3):
                h = self.att(h, mask)  # h 32*32 mask 128*128
        return h

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


# 部分卷积

class PartialConv(layers.Layer):
    def __init__(self, kernel_size=3, strides=1, dilation_rate=1, in_channels=256, padding="same", out_channels=256,
                 use_bias=True, return_mask=True, multi_channel=False, **kwargs):
        super(PartialConv, self).__init__(**kwargs)

        self.multi_channel = multi_channel

        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.return_mask = return_mask
        self.conv = layers.Conv2D(self.out_channels, kernel_size=self.kernel_size, strides=self.strides,
                                  dilation_rate=self.dilation_rate, padding="same")
        if (self.multi_channel):
            self.weight_maskUpdater = tf.ones((self.kernel_size, self.kernel_size, self.in_channels, self.out_channels))
        else:
            self.weight_maskUpdater = tf.ones((self.kernel_size, self.kernel_size, 1, 1))

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[0]

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def build(self, input_shape):
        super(PartialConv, self).build(input_shape)

    def call(self, input, mask=None):

        if (mask is None):
            # 没有mask就创建一个全为1的mask，这样和普通卷积没有什么区别
            if (self.multi_channel):
                mask = tf.ones((input.shape[0], input.shape[1], input.shape[2], input.shape[3]))
            else:
                mask = tf.ones((1, input.shape[1], input.shape[2], 1))

        self.update_mask = tf.nn.conv2d(mask, self.weight_maskUpdater, strides=self.strides, padding="SAME")
        self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
        self.update_mask = tf.clip_by_value(self.update_mask, 0, 1)
        self.mask_ratio = self.mask_ratio * self.update_mask
        # mask and input are similar data type

        raw_output = self.conv(input)
        if (self.use_bias):
            bias_view = self.add_weight(name='kernel',
                                        shape=(self.out_channels,),
                                        initializer='uniform',
                                        trainable=True)
            output = (raw_output - bias_view) * self.mask_ratio + bias_view
            output = output * self.update_mask
        else:
            output = raw_output * self.mask_ratio

        if (self.return_mask):
            return output, self.update_mask
        else:
            return output


# 构造RFR

class RFRNet(models.Model):
    def __init__(self):
        super(RFRNet, self).__init__()

        self.Pconv1 = PartialConv(in_channels=3, out_channels=64, kernel_size=7, strides=2, padding="same",
                                  multi_channel=True, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.Pconv2 = PartialConv(in_channels=64, out_channels=64, kernel_size=7, strides=1, padding="same",
                                  multi_channel=True, use_bias=False)
        self.bn20 = layers.BatchNormalization()
        self.Pconv21 = PartialConv(in_channels=64, out_channels=64, kernel_size=7, strides=1, padding="same",
                                   multi_channel=True, use_bias=False)
        self.Pconv22 = PartialConv(in_channels=64, out_channels=64, kernel_size=7, strides=1, padding="same",
                                   multi_channel=True, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.RFRModule = RFRModule(trainable=True)
        self.Tconv = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.tail1 = PartialConv(in_channels=67, out_channels=32, kernel_size=3, strides=1, padding="same",
                                 multi_channel=True, use_bias=False)
        self.tail2 = Bottleneck(output_dim=32, strides=1)
        self.out = layers.Conv2D(3, kernel_size=3, activation="sigmoid", strides=1, padding="same",use_bias = False)

    def call(self, input, mask):
        # 一次下采样后进行循环推理
        x1, m1 = self.Pconv1(input, mask)
        x1 = tf.nn.relu(self.bn1(x1))
        x1, m1 = self.Pconv2(x1, m1)
        x1 = tf.nn.relu(self.bn20(x1))
        x2 = x1
        x2, m2 = x1, m1
        n, h, w, c = x2.shape
        feature_group = []
        mask_group = []

        self.RFRModule.att.att.att_scores_prev = None
        self.RFRModule.att.att.masks_prev = None

        # 循环推理
        for i in range(6):
            x2, m2 = self.Pconv21(x2, m2)
            x2, m2 = self.Pconv22(x2, m2)
            x2 = tf.nn.leaky_relu(self.bn2(x2))
            x2 = self.RFRModule(x2, m2[..., 0:1])  # 这里的x2 m2都是128*128大小 在RFR模块里dec_3执行后h是32*32大小 需要对mask进行一个resize

            x2 = x2 * m2
            feature_group.append(x2[..., tf.newaxis])
            mask_group.append(m2[..., tf.newaxis])

        x3 = tf.concat(feature_group, axis=-1)
        m3 = tf.concat(mask_group, axis=-1)

        amp_vec = tf.reduce_mean(m3, axis=-1)
        x3 = tf.reduce_mean(x3 * m3, axis=-1) / (amp_vec + 1e-7)
        x3 = tf.reshape(x3, [n, h, w, c])
        m3 = m3[..., -1]

        x4 = self.Tconv(x3)
        x4 = tf.nn.leaky_relu(self.bn3(x4))
        m4 = tf.image.resize(m3, (m3.shape[1] * 2, m3.shape[2] * 2), "bilinear")
        # m4 = layers.UpSampling2D(size = (2,2))(m3)
        x5 = tf.concat([input, x4], axis=-1)  # 这里是c
        m5 = tf.concat([mask, m4], axis=-1)

        x5, _ = self.tail1(x5, m5)
        x5 = tf.nn.leaky_relu(x5)
        x6 = self.tail2(x5)
        x6 = tf.concat([x5, x6], axis=-1)
        output = self.out(x6)

        return output


def getNetwork(log = False):
    inputs = layers.Input(batch_shape=(batch_size, image_size, image_size, image_channel))
    masks = layers.Input(batch_shape=(batch_size, image_size, image_size, mask_channel))
    outputs = RFRNet()(inputs, masks)
    generator = models.Model(inputs=[inputs, masks], outputs=outputs)
    if(log):
        generator.summary()
    return generator
