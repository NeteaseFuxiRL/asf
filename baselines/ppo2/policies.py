import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, dot
from baselines.common.distributions import make_pdtype


def get_nact(ac_space):
    nact = 0
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        nact = ac_space.shape[0]
    elif isinstance(ac_space, spaces.Discrete):
        nact = ac_space.n
    elif isinstance(ac_space, spaces.MultiDiscrete):
        nact = ac_space.nvec
    elif isinstance(ac_space, spaces.MultiBinary):
        nact = ac_space.n
    else:
        raise NotImplementedError
    return nact

def nature_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


def pure_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return h3


class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False, param=None):
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False, param=None):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, param=None):  # pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            pi = fc(h, 'pi', nact, init_scale=0.01)
            vf = fc(h, 'v', 1)[:, 0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnAttentionPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, param=None, reuse=False):  # pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = get_nact(ac_space)
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        state_attention = param
        self.state_attention = state_attention
        actions_onehot = tf.eye(nact, batch_shape=[nbatch])  # [None, num_actions, num_actions]
        batch_actions_onehot = tf.reshape(actions_onehot, shape=(-1, nact))  # [None * num_actions, num_actions]
        with tf.variable_scope("model", reuse=reuse):
            hidden_output_v, hidden_output_p = self.attention_cnn(X, nact, nbatch, state_attention)  # [None, fc_out] [None * num_actions, 512]
            vf = tf.squeeze(tf.identity(fc(hidden_output_v, 'logits_v', 1, init_scale=np.sqrt(2))), axis=[1])
            logits_p_layer_raw = tf.nn.leaky_relu(fc(hidden_output_p, "logits_p_layer_raw", nact, init_scale=np.sqrt(2)))  # [None * num_actions, nact]
            if state_attention not in ["action"]:
                assert logits_p_layer_raw.get_shape().as_list()[1] == nact
                pi = logits_p_layer_raw
            else:
                logits_p_attention = tf.multiply(logits_p_layer_raw, batch_actions_onehot, name="logits_p_attention")
                logits_p_attention_vector = tf.reduce_sum(logits_p_attention, axis=1, name="logits_p_attention_reduce")
                pi = tf.reshape(logits_p_attention_vector, shape=(nbatch, nact), name="logits_p")

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        attention_sigmoid = self.attention

        def get_attention(ob, *_args, **_kwargs):
            attention = sess.run(attention_sigmoid, {X: ob})
            return attention

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.get_attention = get_attention

    def attention_cnn(self, unscaled_images, nact, nbatch, state_attention):
        """
        CNN from Nature paper.
        """
        scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
        activ = tf.nn.relu
        h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
        h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
        h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))  # [None, h, w, c_out]
        h3_flatten = conv_to_fc(h3)  # [None, h * w * c_out]
        hidden_output_v = activ(fc(h3_flatten, 'fc1', 512, init_scale=np.sqrt(2)))  # [None, fc_out]
        conv_result_4d_p = self.create_Channel_Spatial_CBAM(h3, nact, nbatch, state_attention, activ)  # [None * num_actions, h, w, (2 * c) or c]
        flatten_p = conv_to_fc(conv_result_4d_p)  # [None * num_actions, h * w * ((2 * c) or c)]
        hidden_output_p = activ(fc(flatten_p, "p", 512, init_scale=np.sqrt(2)))  # [None * num_actions, 512]
        return hidden_output_v, hidden_output_p

    def create_fc_attention(self, nact, nbatch, state_attention, activ, input_data=None, reduction_ratio=2):
        _input = input_data
        if len(_input.get_shape()) > 2:
            print("Can only process 2d input.")
            return
        _input_dim = _input.get_shape().as_list()[1]
        actions_onehot = tf.eye(nact, batch_shape=[nbatch])
        batch_actions_onehot = tf.reshape(actions_onehot, shape=(-1, nact))
        fc_activ = tf.nn.relu
        if state_attention not in ["action", "action-channel"]:
            action_state_x = _input
        else:
            batch_state_input_attention = tf.reshape(tf.tile(_input, [1, nact]), shape=(-1, _input_dim))
            action_state_x = tf.concat([batch_state_input_attention, batch_actions_onehot], axis=1)
        with tf.variable_scope('fc_attention_reuse', reuse=tf.AUTO_REUSE) as scope:
            state_attention_result = fc_activ(fc(action_state_x, 'fc_attention_hidden', int(_input_dim / reduction_ratio), init_scale=np.sqrt(2)))
            state_attention_result = fc(state_attention_result, 'fc_attention_logit', int(_input_dim), init_scale=np.sqrt(2))
        return state_attention_result

    def get_4d_unit_tensor(self, h, w, nact, nbatch):
        a = np.zeros(shape=(nact, h, w, nact), dtype=np.float32)
        for i in range(nact):
            a[i, :, :, i] = 1.0
        batch_4d_unit = tf.tile(tf.constant(a, dtype=tf.float32), multiples=[nbatch, 1, 1, 1])
        return batch_4d_unit

    def create_Channel_Spatial_CBAM(self, input, nact, nbatch, state_attention, activ):
        '''
        if use action attention mode, different actions will have different attentions from the channel steps
        :param input:
        :param nact:
        :param nbatch:
        :param concat:
        :param state_attention:
        :param activ:
        :return:
        '''
        if len(input.get_shape()) <= 2:
            print("Can only process 4d input with NHWC.")
            return
        n = input.get_shape().as_list()[0]
        h = input.get_shape().as_list()[1]
        w = input.get_shape().as_list()[2]
        c = input.get_shape().as_list()[3]
        reduction_ratio = 2
        max_pool_channel = tf.nn.max_pool(input, ksize=[1, h, w, 1], strides=[1, 1, 1, 1], padding="VALID")  # [None, 1, 1, c]
        max_pool_channel = max_pool_channel[:, 0, 0, :]  # [None, c]
        avg_pool_channel = tf.nn.avg_pool(input, ksize=[1, h, w, 1], strides=[1, 1, 1, 1], padding="VALID")  # [None, 1, 1, c]
        avg_pool_channel = avg_pool_channel[:, 0, 0, :]  # [None, c]
        attention_channel_logits = []
        for pool in [avg_pool_channel, max_pool_channel]:
            attention_channel_logits.append(self.create_fc_attention(nact, nbatch, state_attention, activ, input_data=pool, reduction_ratio=reduction_ratio))
        attention_channel = tf.sigmoid(tf.add(attention_channel_logits[0], attention_channel_logits[1]))  # [nbatch*nact, c]
        attention_channel_expand_dim = tf.expand_dims(attention_channel, 1)  # [nbatch*nact, 1, c]
        attention_channel_expand_dim = tf.expand_dims(attention_channel_expand_dim, 1)  # [nbatch*nact, 1, 1, c]
        assert attention_channel_expand_dim.get_shape().as_list()[1:] == [1, 1, c]
        attention_channel_tile = tf.tile(attention_channel_expand_dim, multiples=[1, h, w, 1])  # [nbatch*nact, h, w, c]
        if state_attention not in ["action"]:
            _input_tile_nhwc = input
        else:
            # tile and adjust the shape of _input
            _input_tile = tf.tile(input, multiples=[1, 1, 1, nact])
            _input_tile_nchw = tf.transpose(_input_tile, [0, 3, 1, 2])
            _input_tile_reshape = tf.reshape(_input_tile_nchw, [-1, c, h, w])
            _input_tile_nhwc = tf.transpose(_input_tile_reshape, [0, 2, 3, 1])
        attention_channel_result = tf.multiply(attention_channel_tile, _input_tile_nhwc)  # [nbatch*nact, h, w, c]
        # begin spatial attention computation
        max_pool_spatial = tf.reduce_max(attention_channel_result, axis=3, keepdims=True)  # [nbatch*nact, h, w, 1]
        avg_pool_spatial = tf.reduce_mean(attention_channel_result, axis=3, keepdims=True)  # [nbatch*nact, h, w, 1]
        attention_spatial_input = tf.concat([max_pool_spatial, avg_pool_spatial], axis=3)  # [nbatch*nact, h, w, 2]
        if state_attention not in ["action"]:
            attention_spatial = tf.sigmoid(conv(attention_spatial_input, "spatial_attention_conv", nf=1, rf=7, stride=1, pad='SAME', init_scale=np.sqrt(2)))  # [nbatch, h, w, 1]
        elif state_attention == "action":
            batch_unit_4d = self.get_4d_unit_tensor(h, w, nact, nbatch)  # [nbatch*nact, h, w, nact]
            attention_spatial_input = tf.concat([attention_spatial_input, batch_unit_4d], axis=3)  # [nbatch*nact, h, w, nact+2]
            attention_spatial = tf.sigmoid(conv(attention_spatial_input, "spatial_attention_conv", nf=1, rf=7, stride=1, pad='SAME', init_scale=np.sqrt(2)))  # [nbatch*nact, h, w, 1]
            assert attention_spatial.get_shape().as_list() == [nbatch * nact, h, w, 1]
        else:
            raise NotImplementedError("Unkonwn attention {}".format(state_attention))
        self.attention = tf.squeeze(attention_spatial)  # [nbatch*nact, h, w]
        attention_spatial_tile = tf.tile(attention_spatial, multiples=[1, 1, 1, c])  # [nbatch*nact, h, w, c]
        if state_attention == "random":
            attention_spatial_tile = tf.random_uniform(shape=[nbatch] + attention_spatial_tile.get_shape().as_list()[1:])
        attention_spatial_result = tf.multiply(attention_spatial_tile, attention_channel_result)
        assert attention_spatial_result.get_shape().as_list()[1:] == [h, w, c]
        assert _input_tile_nhwc.get_shape().as_list()[1:] == [h, w, c]
        attention_result = tf.add(_input_tile_nhwc, attention_spatial_result)  # [None * num_actions, h, w, c]
        return attention_result


class MlpAttentionPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, param=None):  # pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        actions_onehot = tf.eye(actdim, batch_shape=[nbatch])
        batch_actions_onehot = tf.reshape(actions_onehot, shape=(-1, actdim))
        self.attention_size = X.get_shape()[1].value
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            _input_dim = X.get_shape()[1].value
            batch_state_input_attention = tf.reshape(tf.tile(X, [1, actdim]), shape=(-1, _input_dim))
            action_state_x = tf.concat([batch_state_input_attention, batch_actions_onehot], axis=1)
            state_attention_logits = fc(action_state_x, "attentions_output", _input_dim, init_scale=0.01)
            state_attention_prob = tf.nn.softmax(state_attention_logits, axis=1, name="state_attention_result_softmax")
            self.attention_entropy_mean = tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(state_attention_prob, 1e-10, 1.0)) * state_attention_prob, axis=1))
            self.attention_mean, self.attention_std = tf.nn.moments(state_attention_prob, name="soft_std", axes=[1])
            self.mean_attention_mean = tf.reduce_mean(self.attention_mean)
            self.mean_attention_std = tf.reduce_mean(self.attention_std)
            state_attention_prob_expand = state_attention_prob
            fc1 = tf.multiply(state_attention_prob_expand, batch_state_input_attention, name="element_wise_weighted_states")
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            batch_h2 = tf.reshape(tf.tile(h2, [1, actdim]), shape=(-1, 64))
            h2 = tf.concat([batch_h2, fc1], axis=1)
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            pi = tf.reshape(tf.reduce_sum(tf.multiply(pi, batch_actions_onehot), axis=1), shape=(-1, actdim), name='pi_reduce')
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:, 0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        def get_attention(ob, *_args, **_kwargs):
            attention = sess.run(state_attention_prob, {X: ob})
            return attention

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.attention = state_attention_prob
        self.get_attention = get_attention


class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, param=None):  # pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:, 0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                                     initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class MlpStateAttentionPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, param=None):  # pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        self.attention_size = X.get_shape()[1].value
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            _input_dim = X.get_shape()[1].value
            state_attention_logits = fc(X, "attentions_output", _input_dim, init_scale=0.01)
            state_attention_prob = tf.nn.softmax(state_attention_logits, axis=1, name="state_attention_result_softmax")
            self.attention_entropy_mean = tf.reduce_mean(tf.reduce_sum(tf.log(tf.clip_by_value(state_attention_prob, 1e-10, 1.0)) * state_attention_prob, axis=1))
            self.attention_mean, self.attention_std = tf.nn.moments(state_attention_prob, name="soft_std", axes=[1])
            self.mean_attention_mean = tf.reduce_mean(self.attention_mean)
            self.mean_attention_std = tf.reduce_mean(self.attention_std)
            state_attention_prob_expand = state_attention_prob
            fc1 = tf.multiply(state_attention_prob_expand, X, name="element_wise_weighted_states")
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            h2 = tf.concat([h2, fc1], axis=1)
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:, 0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                                     initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
