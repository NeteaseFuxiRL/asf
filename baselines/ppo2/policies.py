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
            hidden_output_v, hidden_output_p = self.attention_cnn(X, nact, nbatch)  # [nbatch, fc_out] [nbatch * num_actions, 512]
            # vf = tf.squeeze(tf.identity(fc(hidden_output_v, 'logits_v', 1, init_scale=np.sqrt(2))), axis=[1])
            vf = fc(hidden_output_v, 'v', 1)[:, 0]
            logits_p_layer_raw = fc(hidden_output_p, "logits_p_layer_raw", nact, init_scale=0.01)  # [nbatch * num_actions, num_actions]
            if self.state_attention not in ["action"]:
                assert logits_p_layer_raw.get_shape().as_list() == [nbatch, nact]
                pi = logits_p_layer_raw
            else:
                logits_p_attention = tf.multiply(logits_p_layer_raw, batch_actions_onehot, name="logits_p_attention")  # [nbatch * num_actions, num_actions]
                logits_p_attention_vector = tf.reduce_sum(logits_p_attention, axis=1, name="logits_p_attention_reduce")  # [nbatch * num_actions]
                pi = tf.reshape(logits_p_attention_vector, shape=(nbatch, nact), name="logits_p")  # [nbatch , num_actions,]

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

        def get_attention(ob, *_args, **_kwargs):
            attention = sess.run(self.output_attention, {X: ob})
            return attention

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.get_attention = get_attention

    def attention_cnn(self, unscaled_images, nact, nbatch):
        scaled_images = tf.cast(unscaled_images, tf.float32) / 255.  # [None, 84, 84, 1]
        activ = tf.nn.relu
        h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
        h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
        h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))  # [nbatch, h, w, c]
        h = h3.get_shape().as_list()[1]
        w = h3.get_shape().as_list()[2]
        c = h3.get_shape().as_list()[3]
        h3_flatten = conv_to_fc(h3)
        fc_out_unit = 512
        if self.state_attention == "state":
            batch_state_input_attention = h3_flatten  # [nbatch, h*w*c]
            batch_actions_onehot = batch_state_input_attention  # [nbatch, h*w*c]
            attention_input = batch_state_input_attention
        else:
            # expand the state
            batch_state_input_attention = tf.reshape(tf.tile(h3_flatten, [1, nact]), shape=(-1, h * w * c))  # [nbatch*nact,*h*w*c]
            # expand the action
            actions_onehot = tf.eye(nact, batch_shape=[nbatch])
            batch_actions_onehot = tf.reshape(actions_onehot, shape=(-1, nact))  # [nbatch*nact, nact]
            # combine the action and the state
            attention_input = tf.concat([batch_state_input_attention, dot(batch_actions_onehot, 'attention_query_dot', h * w * c, init_scale=np.sqrt(2))], axis=1)  # [nbatch*nact, 2*h*w*c]
        # compute the attention logits
        attention_logits = fc(attention_input, "attention_logits", h * w, init_scale=np.sqrt(2))  # [nbatch*nact, h*w] or [nbatch, h*w]
        # get the final attention
        self.attention = tf.sigmoid(attention_logits)  # [nbatch*nact, h*w] or [nbatch, h*w]
        # multiply the attention weights with the original state
        if self.state_attention == "state":
            attention_reshape = tf.tile(tf.reshape(self.attention, shape=(nbatch * h * w, 1)), [1, c])  # [nbatch*h*w,c]
            raw_attended_result = tf.reshape(tf.multiply(tf.reshape(h3, shape=(nbatch * h * w, c)), attention_reshape), shape=(nbatch, h * w * c))  # [nbatch, h*w*c]
            self.attention_loss = tf.constant(value=0, dtype="float32")
        else:
            h3_flatten_tile = tf.reshape(tf.tile(h3_flatten, [1, nact]), shape=(-1, c))  # [nbatch*nact*h*w, c]
            attention_reshape = tf.tile(tf.reshape(self.attention, shape=(nbatch * nact * h * w, 1)), [1, c])  # [nbatch*nact*h*w*, c]
            raw_attended_result = tf.reshape(tf.multiply(h3_flatten_tile, attention_reshape), shape=(nbatch * nact, h * w * c))  # [nbatch*nact, h*w*c]
            # reshape the attention matrix for attention image print
            self.output_attention = tf.reshape(tf.expand_dims(self.attention, dim=-1), shape=(nbatch * nact, h, w, 1))
            # compute the attention differences for loss
            attention4loss = tf.reshape(self.attention, shape=(nbatch, nact, h * w))  # [nbatch, nact, h*w]
            attention_loss = 0
            for i in range(1, nact):
                attention_loss = attention_loss + tf.reduce_sum(tf.square(attention4loss - tf.concat([attention4loss[:, :i, :], attention4loss[:, :-i, :]], axis=1)), axis=[1, 2])
            print("attention loss shape is {}, batch size is {}".format(attention_loss.get_shape().as_list(), nbatch))
            self.attention_loss = tf.exp(-tf.reduce_mean(attention_loss, axis=0))
        # combine the attended states with the original states
        attention_result_in_batch = tf.concat([batch_state_input_attention, raw_attended_result], axis=-1)
        # use an fc for hidden output p
        hidden_output_p = activ(fc(attention_result_in_batch, "p", fc_out_unit, init_scale=np.sqrt(2)))  # [nbatch*nact, fc_out_unit], n_params = c*fc_out_unit+fc_out_unit
        # assert hidden_output_p.get_shape().as_list()[0] == nbatch * nact, "Tensor shape is {}, right shape is {}".format(hidden_output_p.get_shape().as_list(), [nbatch * nact, fc_out_unit])

        # compute the hidden value for v as usual
        h3_flatten = conv_to_fc(h3)  # [None, h * w * c_out]
        hidden_output_v = activ(fc(h3_flatten, 'fc1', fc_out_unit, init_scale=np.sqrt(2)))  # [None, fc_out], n_params = h * w * c* 512+512
        return hidden_output_v, hidden_output_p



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
