
import tensorflow.keras.layers as layers
import tensorflow as tf

class KnowledgeConsistentAttention(layers.Layer):

    def __init__(self, patch_size=3, propagate_size=3, stride=1, output_dim=256, **kwargs):
        self.output_dim = output_dim
        super(KnowledgeConsistentAttention, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.prop_kernels = None
        self.att_scores_prev = None
        self.masks_prev = None
        self.ratio = tf.ones(1)

    def build(self, input_shape):
        super(KnowledgeConsistentAttention, self).build(input_shape)

    def call(self, foreground, masks):

        bz, w, h, nc = foreground.shape
        # masks shape == foreground.shape
        if (masks.shape[1] != h):
            masks = tf.image.resize(masks, (h, w), "bilinear")

        background = foreground[:]

        conv_kernels_all = tf.transpose(background, [0, 3, 1, 2])
        conv_kernels_all = tf.reshape(conv_kernels_all, [bz, nc, h * w, 1, 1])  # 他这直接单个特征通道直接进行相似度计算，感觉误差会很大啊
        conv_kernels_all = tf.transpose(conv_kernels_all, [0, 3, 4, 1, 2])  # b k k c hw

        output_tensor = []
        att_score = []
        for i in range(bz):
            feature_map = foreground[i:i + 1]
            conv_kernels = conv_kernels_all[i] + 0.0000001  # k k c hw
            norm_factor = tf.reduce_sum(conv_kernels ** 2, [0, 1, 2], keepdims=True) ** 0.5
            conv_kernels = conv_kernels / norm_factor
            conv_result = tf.nn.conv2d(feature_map, conv_kernels, strides=[1, 1, 1, 1], padding="SAME")  # 1 h w hw
            if (self.propagate_size != 1):
                if (self.prop_kernels is None):
                    self.prop_kernels = tf.ones([conv_result.shape[-1], 1, self.propagate_size, self.propagate_size])
                    self.prop_kernels.requires_grad = False

                conv_result = tf.nn.avg_pool2d(conv_result, 3, 1, padding="SAME") * 9
            attention_scores = tf.nn.softmax(conv_result, axis=-1)

            if (self.att_scores_prev is not None):
                attention_scores = (self.att_scores_prev[i:i + 1] * self.masks_prev[i:i + 1] + attention_scores * (
                            tf.abs(self.ratio) + 1e-7)) / (self.masks_prev[i:i + 1] + (tf.abs(self.ratio) + 1e-7))

            att_score.append(attention_scores)
            feature_map = tf.nn.conv2d_transpose(attention_scores, conv_kernels, output_shape=[1, w, h, nc], strides=1,
                                                 padding="SAME")
            final_output = feature_map
            output_tensor.append(final_output)
        self.att_scores_prev = tf.reshape(tf.concat(att_score, axis=0), [bz, h, w, h * w])
        self.masks_prev = tf.reshape(masks, [bz, h, w, 1])
        return tf.concat(output_tensor, axis=0)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class AttentionModule(layers.Layer):

    def __init__(self, inchannel, patch_size_list=[1], propagate_size_list=[3], stride_list=[1], **kwargs):
        super(AttentionModule, self).__init__(**kwargs)
        self.att = KnowledgeConsistentAttention(patch_size_list[0], propagate_size_list[0], stride_list[0])
        self.num_of_modules = len(patch_size_list)
        self.combiner = layers.Conv2D(inchannel, kernel_size=1, padding="same")

    def build(self, input_shape):
        super(AttentionModule, self).build(input_shape)

    def call(self, foreground, mask):
        outputs = self.att(foreground, mask)
        outputs = tf.concat([outputs, foreground], axis=-1)
        outputs = self.combiner(outputs)
        return outputs