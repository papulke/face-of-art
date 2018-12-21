import tensorflow as tf
import numpy as np


def TPS(U, nx, ny, cp, out_size):
    """Thin Plate Spline Spatial Transformer Layer
    TPS control points are arranged in a regular grid.

    U : float Tensor
        shape [num_batch, height, width, num_channels].
    nx : int
        The number of control points on x-axis
    ny : int
        The number of control points on y-axis
    cp : float Tensor
        control points. shape [num_batch, nx*ny, 2].
    out_size: tuple of two ints
        The size of the output of the network (height, width)

    ----------
    Reference :
      https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
    """

    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        channels = tf.shape(im)[3]

        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return output

    def _meshgrid(height, width, fp):
        x_t = tf.matmul(
            tf.ones(shape=tf.stack([height, 1])),
            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(
            tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
            tf.ones(shape=tf.stack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        x_t_flat_b = tf.expand_dims(x_t_flat, 0)  # [1, 1, h*w]
        y_t_flat_b = tf.expand_dims(y_t_flat, 0)  # [1, 1, h*w]

        num_batch = tf.shape(fp)[0]
        px = tf.expand_dims(fp[:, :, 0], 2)  # [n, nx*ny, 1]
        py = tf.expand_dims(fp[:, :, 1], 2)  # [n, nx*ny, 1]
        d = tf.sqrt(tf.pow(x_t_flat_b - px, 2.) + tf.pow(y_t_flat_b - py, 2.))
        r = tf.pow(d, 2) * tf.log(d + 1e-6)  # [n, nx*ny, h*w]
        x_t_flat_g = tf.tile(x_t_flat_b, tf.stack([num_batch, 1, 1]))  # [n, 1, h*w]
        y_t_flat_g = tf.tile(y_t_flat_b, tf.stack([num_batch, 1, 1]))  # [n, 1, h*w]
        ones = tf.ones_like(x_t_flat_g)  # [n, 1, h*w]

        grid = tf.concat([ones, x_t_flat_g, y_t_flat_g, r], 1)  # [n, nx*ny+3, h*w]
        return grid

    def _transform(T, fp, input_dim, out_size):
        num_batch = tf.shape(input_dim)[0]
        height = input_dim.shape[1]
        width = input_dim.shape[2]
        num_channels = input_dim.shape[3]

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        grid = _meshgrid(out_height, out_width, fp)  # [2, h*w]

        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        T_g = tf.matmul(T, grid)  # MARK
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tf.reshape(x_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])

        input_transformed = _interpolate(
            input_dim, x_s_flat, y_s_flat, out_size)

        output = tf.reshape(
            input_transformed,
            tf.stack([num_batch, out_height, out_width, num_channels]))
        return output

    def _solve_system(cp, nx, ny):
        gx = 2. / nx  # grid x size
        gy = 2. / ny  # grid y size
        cx = -1. + gx / 2.  # x coordinate
        cy = -1. + gy / 2.  # y coordinate

        p_ = np.empty([nx * ny, 3], dtype='float32')
        i = 0
        #yael: generate regular grid points, source points
        for _ in range(ny):
            for _ in range(nx):
                p_[i, :] = 1, cx, cy
                i += 1
                cx += gx
            cx = -1. + gx / 2
            cy += gy

        p_1 = p_.reshape([nx * ny, 1, 3])
        p_2 = p_.reshape([1, nx * ny, 3])
        #calculate weights according to distance between points
        d = np.sqrt(np.sum((p_1 - p_2) ** 2, 2))  # [nx*ny, nx*ny]
        r = d * d * np.log(d * d + 1e-5)
        W = np.zeros([nx * ny + 3, nx * ny + 3], dtype='float32')
        W[:nx * ny, 3:] = r
        W[:nx * ny, :3] = p_
        W[nx * ny:, 3:] = p_.T

        num_batch = tf.shape(cp)[0]
        fp = tf.constant(p_[:, 1:], dtype='float32')  # [nx*ny, 2]
        fp = tf.expand_dims(fp, 0)  # [1, nx*ny, 2]
        fp = tf.tile(fp, tf.stack([num_batch, 1, 1]))  # [n, nx*ny, 2]

        W_inv = np.linalg.inv(W)
        W_inv_t = tf.constant(W_inv, dtype='float32')  # [nx*ny+3, nx*ny+3]
        W_inv_t = tf.expand_dims(W_inv_t, 0)  # [1, nx*ny+3, nx*ny+3]
        W_inv_t = tf.tile(W_inv_t, tf.stack([num_batch, 1, 1]))

        cp_pad = tf.pad(cp + fp, [[0, 0], [0, 3], [0, 0]], "CONSTANT")
        T = tf.matmul(W_inv_t, cp_pad)
        T = tf.transpose(T, [0, 2, 1])

        return T, fp

    cp = tf.reshape(cp, [tf.shape(U)[0], nx*ny, 2])

    T, fp = _solve_system(cp, nx, ny)
    output = _transform(T, fp, U, out_size)
    return output
