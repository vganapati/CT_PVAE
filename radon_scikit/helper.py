from torchvision import datasets
import numpy as np
import os
from warnings import warn
import math
from abc import ABC
import textwrap
import sys
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from _warps_cy import _warp_fast

def warp(image, inverse_map, map_args=None, output_shape=None, order=None,
         mode='constant', cval=0., clip=True, preserve_range=False):

    if map_args is None:
        map_args = {}

    if image.size == 0:
        raise ValueError("Cannot warp empty image with dimensions",
                         image.shape)

    order = _validate_interpolation_order(image.dtype, order)

    if order > 0:
        image = convert_to_float(image, preserve_range)
        if image.dtype == np.float16:
            image = image.astype(np.float32)

    input_shape = np.array(image.shape)

    if output_shape is None:
        output_shape = input_shape
    else:
        output_shape = safe_as_int(output_shape)

    warped = None

    if order == 2:
        # When fixing this issue, make sure to fix the branches further
        # below in this function
        warn("Bi-quadratic interpolation behavior has changed due "
             "to a bug in the implementation of scikit-image. "
             "The new version now serves as a wrapper "
             "around SciPy's interpolation functions, which itself "
             "is not verified to be a correct implementation. Until "
             "skimage's implementation is fixed, we recommend "
             "to use bi-linear or bi-cubic interpolation instead.")

    if order in (1, 3) and not map_args:
        # use fast Cython version for specific interpolation orders and input

        matrix = None

        if isinstance(inverse_map, np.ndarray) and inverse_map.shape == (3, 3):
            # inverse_map is a transformation matrix as numpy array
            matrix = inverse_map

        elif isinstance(inverse_map, HOMOGRAPHY_TRANSFORMS):
            # inverse_map is a homography
            matrix = inverse_map.params

        elif (hasattr(inverse_map, '__name__') and
              inverse_map.__name__ == 'inverse' and
              get_bound_method_class(inverse_map) in HOMOGRAPHY_TRANSFORMS):
            # inverse_map is the inverse of a homography
            matrix = np.linalg.inv(inverse_map.__self__.params)

        if matrix is not None:
            matrix = matrix.astype(image.dtype)
            ctype = 'float32_t' if image.dtype == np.float32 else 'float64_t'
            if image.ndim == 2:
                warped = _warp_fast(image, matrix,
                                           output_shape=output_shape,
                                           order=order, mode=mode, cval=cval) # _warp_fast[ctype]
            elif image.ndim == 3:
                dims = []
                for dim in range(image.shape[2]):
                    dims.append(_warp_fast(image[..., dim], matrix,
                                                  output_shape=output_shape,
                                                  order=order, mode=mode,
                                                  cval=cval))
                warped = np.dstack(dims)

    if warped is None:
        # use ndi.map_coordinates

        if (isinstance(inverse_map, np.ndarray) and
                inverse_map.shape == (3, 3)):
            # inverse_map is a transformation matrix as numpy array,
            # this is only used for order >= 4.
            inverse_map = ProjectiveTransform(matrix=inverse_map)

        if isinstance(inverse_map, np.ndarray):
            # inverse_map is directly given as coordinates
            coords = inverse_map
        else:
            # inverse_map is given as function, that transforms (N, 2)
            # destination coordinates to their corresponding source
            # coordinates. This is only supported for 2(+1)-D images.

            if image.ndim < 2 or image.ndim > 3:
                raise ValueError("Only 2-D images (grayscale or color) are "
                                 "supported, when providing a callable "
                                 "`inverse_map`.")

            def coord_map(*args):
                return inverse_map(*args, **map_args)

            if len(input_shape) == 3 and len(output_shape) == 2:
                # Input image is 2D and has color channel, but output_shape is
                # given for 2-D images. Automatically add the color channel
                # dimensionality.
                output_shape = (output_shape[0], output_shape[1],
                                input_shape[2])

            coords = warp_coords(coord_map, output_shape)

        # Pre-filtering not necessary for order 0, 1 interpolation
        prefilter = order > 1

        ndi_mode = _to_ndimage_mode(mode)
        warped = ndi.map_coordinates(image, coords, prefilter=prefilter,
                                     mode=ndi_mode, order=order, cval=cval)

    _clip_warp_output(image, warped, mode, cval, clip)

    return warped

# def _transform_metric(x, y, H, x_, y_):
#     temp_x = H[0, 0] * x + H[0, 2]
#     temp_y = H[2, 0] * y + H[2, 2]
#     x_[:] = np.asarray(temp_x)
#     y_[:] = np.asarray(temp_y)

# def _transform_affine(x, y, H, x_, y_):
#     x_[0] = H[0] * x + H[1] * y + H[2]
#     y_[0] = H[3] * x + H[4] * y + H[5]

# def _transform_projective(x, y, H, x_, y_):
#     z_ = H[6] * x + H[7] * y + H[8]
#     x_[0] = (H[0] * x + H[1] * y + H[2]) / z_
#     y_[0] = (H[3] * x + H[4] * y + H[5]) / z_

# def _warp_fast(image, H, output_shape=None, order=1, mode='constant', cval=0):
#     img = np.ascontiguousarray(image)
#     M = np.ascontiguousarray(H)

#     if mode not in ('constant', 'wrap', 'symmetric', 'reflect', 'edge'):
#         raise ValueError("Invalid mode specified. Please use `constant`, "
#                          "`edge`, `wrap`, `reflect` or `symmetric`.")

#     if output_shape is None:
#         out_r = img.shape[0]
#         out_c = img.shape[1]
#     else:
#         out_r = output_shape[0]
#         out_c = output_shape[1]

#     out = np.zeros((out_r, out_c), dtype=img.dtype)

#     def transform_func(x, y, H, x_, y_):
#         if H[2, 0] == 0 and H[2, 1] == 0 and H[2, 2] == 1:
#             if H[0, 1] == 0 and H[1, 0] == 0:
#                 _transform_metric(x, y, H, x_, y_)
#             else:
#                 _transform_affine(x, y, H, x_, y_)
#         else:
#             _transform_projective(x, y, H, x_, y_)

#     def interp_func(input_array, rows, cols, r, c, mode, cval):
#         return map_coordinates(input_array, [[r]], [[c]], order=order, mode=mode, cval=cval)[0]

#     for tfr in range(out_r):
#         for tfc in range(out_c):
#             c, r = np.float64(tfc), np.float64(tfr)
#             transform_func(c, r, M, c, r)
#             out[tfr, tfc] = interp_func(img, img.shape[0], img.shape[1], r, c, mode, cval)

#     return out

###################################################################################

def _convert(image, dtype, force_copy=False, uniform=False):
    image = np.asarray(image)
    dtypeobj_in = image.dtype
    if dtype is np.floating:
        dtypeobj_out = np.dtype('float64')
    else:
        dtypeobj_out = np.dtype(dtype)
    dtype_in = dtypeobj_in.type
    dtype_out = dtypeobj_out.type
    kind_in = dtypeobj_in.kind
    kind_out = dtypeobj_out.kind
    itemsize_in = dtypeobj_in.itemsize
    itemsize_out = dtypeobj_out.itemsize

    # Below, we do an `issubdtype` check.  Its purpose is to find out
    # whether we can get away without doing any image conversion.  This happens
    # when:
    #
    # - the output and input dtypes are the same or
    # - when the output is specified as a type, and the input dtype
    #   is a subclass of that type (e.g. `np.floating` will allow
    #   `float32` and `float64` arrays through)

    if np.issubdtype(dtype_in, np.obj2sctype(dtype)):
        if force_copy:
            image = image.copy()
        return image

    if not (dtype_in in _supported_types and dtype_out in _supported_types):
        raise ValueError(f'Cannot convert from {dtypeobj_in} to '
                         f'{dtypeobj_out}.')

    if kind_in in 'ui':
        imin_in = np.iinfo(dtype_in).min
        imax_in = np.iinfo(dtype_in).max
    if kind_out in 'ui':
        imin_out = np.iinfo(dtype_out).min
        imax_out = np.iinfo(dtype_out).max

    # any -> binary
    if kind_out == 'b':
        return image > dtype_in(dtype_range[dtype_in][1] / 2)

    # binary -> any
    if kind_in == 'b':
        result = image.astype(dtype_out)
        if kind_out != 'f':
            result *= dtype_out(dtype_range[dtype_out][1])
        return result

    # float -> any
    if kind_in == 'f':
        if kind_out == 'f':
            # float -> float
            return image.astype(dtype_out)

        if np.min(image) < -1.0 or np.max(image) > 1.0:
            raise ValueError("Images of type float must be between -1 and 1.")
        # floating point -> integer
        # use float type that can represent output integer type
        computation_type = _dtype_itemsize(itemsize_out, dtype_in,
                                           np.float32, np.float64)

        if not uniform:
            if kind_out == 'u':
                image_out = np.multiply(image, imax_out,
                                        dtype=computation_type)
            else:
                image_out = np.multiply(image, (imax_out - imin_out) / 2,
                                        dtype=computation_type)
                image_out -= 1.0 / 2.
            np.rint(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)
        elif kind_out == 'u':
            image_out = np.multiply(image, imax_out + 1,
                                    dtype=computation_type)
            np.clip(image_out, 0, imax_out, out=image_out)
        else:
            image_out = np.multiply(image, (imax_out - imin_out + 1.0) / 2.0,
                                    dtype=computation_type)
            np.floor(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)
        return image_out.astype(dtype_out)

    # signed/unsigned int -> float
    if kind_out == 'f':
        # use float type that can exactly represent input integers
        computation_type = _dtype_itemsize(itemsize_in, dtype_out,
                                           np.float32, np.float64)

        if kind_in == 'u':
            # using np.divide or np.multiply doesn't copy the data
            # until the computation time
            image = np.multiply(image, 1. / imax_in,
                                dtype=computation_type)
            # DirectX uses this conversion also for signed ints
            # if imin_in:
            #     np.maximum(image, -1.0, out=image)
        elif kind_in == 'i':
            # From DirectX conversions:
            # The most negative value maps to -1.0f
            # Every other value is converted to a float (call it c)
            # and then result = c * (1.0f / (2⁽ⁿ⁻¹⁾-1)).

            image = np.multiply(image, 1. / imax_in,
                                dtype=computation_type)
            np.maximum(image, -1.0, out=image)

        else:
            image = np.add(image, 0.5, dtype=computation_type)
            image *= 2 / (imax_in - imin_in)

        return np.asarray(image, dtype_out)

    # unsigned int -> signed/unsigned int
    if kind_in == 'u':
        if kind_out == 'i':
            # unsigned int -> signed int
            image = _scale(image, 8 * itemsize_in, 8 * itemsize_out - 1)
            return image.view(dtype_out)
        else:
            # unsigned int -> unsigned int
            return _scale(image, 8 * itemsize_in, 8 * itemsize_out)

    # signed int -> unsigned int
    if kind_out == 'u':
        image = _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out)
        result = np.empty(image.shape, dtype_out)
        np.maximum(image, 0, out=result, dtype=image.dtype, casting='unsafe')
        return result

    # signed int -> signed int
    if itemsize_in > itemsize_out:
        return _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out - 1)

    image = image.astype(_dtype_bits('i', itemsize_out * 8))
    image -= imin_in
    image = _scale(image, 8 * itemsize_in, 8 * itemsize_out, copy=False)
    image += imin_out
    return image.astype(dtype_out)

def _dtype_itemsize(itemsize, *dtypes):

    return next(dt for dt in dtypes if np.dtype(dt).itemsize >= itemsize)

def _scale(a, n, m, copy=True):

    kind = a.dtype.kind
    if n > m and a.max() < 2 ** m:
        mnew = int(np.ceil(m / 2) * 2)
        if mnew > m:
            dtype = f'int{mnew}'
        else:
            dtype = f'uint{mnew}'
        n = int(np.ceil(n / 2) * 2)
        warn(f'Downcasting {a.dtype} to {dtype} without scaling because max '
             f'value {a.max()} fits in {dtype}',
             stacklevel=3)
        return a.astype(_dtype_bits(kind, m))
    elif n == m:
        return a.copy() if copy else a
    elif n > m:
        # downscale with precision loss
        if copy:
            b = np.empty(a.shape, _dtype_bits(kind, m))
            np.floor_divide(a, 2**(n - m), out=b, dtype=a.dtype,
                            casting='unsafe')
            return b
        else:
            a //= 2**(n - m)
            return a
    elif m % n == 0:
        # exact upscale to a multiple of `n` bits
        if copy:
            b = np.empty(a.shape, _dtype_bits(kind, m))
            np.multiply(a, (2**m - 1) // (2**n - 1), out=b, dtype=b.dtype)
            return b
        else:
            a = a.astype(_dtype_bits(kind, m, a.dtype.itemsize), copy=False)
            a *= (2**m - 1) // (2**n - 1)
            return a
    else:
        # upscale to a multiple of `n` bits,
        # then downscale with precision loss
        o = (m // n + 1) * n
        if copy:
            b = np.empty(a.shape, _dtype_bits(kind, o))
            np.multiply(a, (2**o - 1) // (2**n - 1), out=b, dtype=b.dtype)
            b //= 2**(o - m)
            return b
        else:
            a = a.astype(_dtype_bits(kind, o, a.dtype.itemsize), copy=False)
            a *= (2**o - 1) // (2**n - 1)
            a //= 2**(o - m)
            return a

dtype_range = {bool: (False, True),
               np.bool_: (False, True),
               float: (-1, 1),
               np.float_: (-1, 1),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}
_supported_types = list(dtype_range.keys())

def _dtype_bits(kind, bits, itemsize=1):
    s = next(i for i in (itemsize, ) + (2, 4, 8) if
             bits < (i * 8) or (bits == (i * 8) and kind == 'u'))

    return np.dtype(kind + str(s))

def _fix_ndimage_mode(mode):
    # SciPy 1.6.0 introduced grid variants of constant and wrap which
    # have less surprising behavior for images. Use these when available
    grid_modes = {'constant': 'grid-constant', 'wrap': 'grid-wrap'}
    return grid_modes.get(mode, mode)

def _to_ndimage_mode(mode):
    """Convert from `numpy.pad` mode name to the corresponding ndimage mode."""
    mode_translation_dict = dict(constant='constant', edge='nearest',
                                 symmetric='reflect', reflect='mirror',
                                 wrap='wrap')
    if mode not in mode_translation_dict:
        raise ValueError(
            f"Unknown mode: '{mode}', or cannot translate mode. The "
             f"mode should be one of 'constant', 'edge', 'symmetric', "
             f"'reflect', or 'wrap'. See the documentation of numpy.pad for "
             f"more info.")
    return _fix_ndimage_mode(mode_translation_dict[mode])

def _clip_warp_output(input_image, output_image, mode, cval, clip):
    if clip:
        min_val = np.min(input_image)
        if np.isnan(min_val):
            # NaNs detected, use NaN-safe min/max
            min_func = np.nanmin
            max_func = np.nanmax
            min_val = min_func(input_image)
        else:
            min_func = np.min
            max_func = np.max
        max_val = max_func(input_image)

        # Check if cval has been used such that it expands the effective input
        # range
        preserve_cval = (mode == 'constant'
                         and not min_val <= cval <= max_val
                         and min_func(output_image) <= cval <= max_func(output_image))

        # expand min/max range to account for cval
        if preserve_cval:
            # cast cval to the same dtype as the input image
            cval = input_image.dtype.type(cval)
            min_val = min(min_val, cval)
            max_val = max(max_val, cval)

        np.clip(output_image, min_val, max_val, out=output_image)

def _stackcopy(a, b):

    if a.ndim == 3:
        a[:] = b[:, :, np.newaxis]
    else:
        a[:] = b

def warp_coords(coord_map, shape, dtype=np.float64):

    shape = safe_as_int(shape)
    rows, cols = shape[0], shape[1]
    coords_shape = [len(shape), rows, cols]
    if len(shape) == 3:
        coords_shape.append(shape[2])
    coords = np.empty(coords_shape, dtype=dtype)

    # Reshape grid coordinates into a (P, 2) array of (row, col) pairs
    tf_coords = np.indices((cols, rows), dtype=dtype).reshape(2, -1).T

    # Map each (row, col) pair to the source image according to
    # the user-provided mapping
    tf_coords = coord_map(tf_coords)

    # Reshape back to a (2, M, N) coordinate grid
    tf_coords = tf_coords.T.reshape((-1, cols, rows)).swapaxes(1, 2)

    # Place the y-coordinate mapping
    _stackcopy(coords[1, ...], tf_coords[0, ...])

    # Place the x-coordinate mapping
    _stackcopy(coords[0, ...], tf_coords[1, ...])

    if len(shape) == 3:
        coords[2, ...] = range(shape[2])

    return coords

def get_bound_method_class(m):
    return m.im_class if sys.version < '3' else m.__self__.__class__

def img_as_float(image, force_copy=False):
    return _convert(image, np.floating, force_copy)

def convert_to_float(image, preserve_range):
    if image.dtype == np.float16:
        return image.astype(np.float32)
    if preserve_range:
        # Convert image to double only if it is not single or double
        # precision float
        if image.dtype.char not in 'df':
            image = image.astype(float)
    else:
        image = img_as_float(image)
    return image

def get_images(img_type = 'mnist'):
    if img_type == 'mnist':
        mnist = datasets.MNIST(root='./data', train=True, download=True)
        x_train = mnist.data.float() / 255.0  # Normalize pixel values to the range [0, 1]
    else:
        x_train = np.load(img_type + '_training.npy')
    return(x_train)

def create_folder(save_path=None,**kwargs):
    try: 
        os.makedirs(save_path)
    except OSError:
        if not os.path.isdir(save_path):
            raise

def _validate_interpolation_order(image_dtype, order):
    if order is None:
        return 0 if image_dtype == bool else 1

    if order < 0 or order > 5:
        raise ValueError("Spline interpolation order has to be in the "
                         "range 0-5.")

    if image_dtype == bool and order != 0:
        raise ValueError(
            "Input image dtype is bool. Interpolation is not defined "
            "with bool data type. Please set order to 0 or explicitly "
            "cast input image to another data type.")

    return order

def safe_as_int(val, atol=1e-3):
    mod = np.asarray(val) % 1                # Extract mantissa

    # Check for and subtract any mod values > 0.5 from 1
    if mod.ndim == 0:                        # Scalar input, cannot be indexed
        if mod > 0.5:
            mod = 1 - mod
    else:                                    # Iterable input, now ndarray
        mod[mod > 0.5] = 1 - mod[mod > 0.5]  # Test on each side of nearest int

    try:
        np.testing.assert_allclose(mod, 0, atol=atol)
    except AssertionError:
        raise ValueError(f'Integer argument required but received '
                         f'{val}, check inputs.')

    return np.round(val).astype(np.int64)

def _center_and_normalize_points(points):
 
    n, d = points.shape
    centroid = np.mean(points, axis=0)

    centered = points - centroid
    rms = np.sqrt(np.sum(centered ** 2) / n)

    # if all the points are the same, the transformation matrix cannot be
    # created. We return an equivalent matrix with np.nans as sentinel values.
    # This obviates the need for try/except blocks in functions calling this
    # one, and those are only needed when actual 0 is reached, rather than some
    # small value; ie, we don't need to worry about numerical stability here,
    # only actual 0.
    if rms == 0:
        return np.full((d + 1, d + 1), np.nan), np.full_like(points, np.nan)

    norm_factor = np.sqrt(d) / rms

    part_matrix = norm_factor * np.concatenate(
            (np.eye(d), -centroid[:, np.newaxis]), axis=1
            )
    matrix = np.concatenate(
            (part_matrix, [[0,] * d + [1]]), axis=0
            )

    points_h = np.row_stack([points.T, np.ones(n)])

    new_points_h = (matrix @ points_h).T

    new_points = new_points_h[:, :d]
    new_points /= new_points_h[:, d:]

    return matrix, new_points

def _umeyama(src, dst, estimate_scale):

    src = np.asarray(src)
    dst = np.asarray(dst)

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T

def _euler_rotation(axis, angle):
    i = axis
    s = (-1)**i * np.sin(angle)
    c = np.cos(angle)
    R2 = np.array([[c, -s],
                   [s,  c]])
    Ri = np.eye(3)
    # We need the axes other than the rotation axis, in the right order:
    # 0 -> (1, 2); 1 -> (0, 2); 2 -> (0, 1).
    axes = sorted({0, 1, 2} - {axis})
    # We then embed the 2-axis rotation matrix into the full matrix.
    # (1, 2) -> R[1:3:1, 1:3:1] = R2, (0, 2) -> R[0:3:2, 0:3:2] = R2, etc.
    sl = slice(axes[0], axes[1] + 1, axes[1] - axes[0])
    Ri[sl, sl] = R2
    return Ri

def _euler_rotation_matrix(angles, axes=None):
    if axes is None:
        axes = range(3)
    dim = len(angles)
    R = np.eye(dim)
    for i, angle in zip(axes, angles):
        R = R @ _euler_rotation(i, angle)
    return R

class _GeometricTransform(ABC):
    """Abstract base class for geometric transformations."""

    def __call__(self, coords):
         """Apply forward transformation.

        Parameters
        ----------
        coords : (N, 2) array_like
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Destination coordinates.

        """

    def inverse(self):
        """Return a transform object representing the inverse."""

    def residuals(self, src, dst):
        return np.sqrt(np.sum((self(src) - dst)**2, axis=1))

class ProjectiveTransform(_GeometricTransform):

    def __init__(self, matrix=None, *, dimensionality=2):
        if matrix is None:
            # default to an identity transform
            matrix = np.eye(dimensionality + 1)
        else:
            matrix = np.asarray(matrix)
            dimensionality = matrix.shape[0] - 1
            if matrix.shape != (dimensionality + 1, dimensionality + 1):
                raise ValueError("invalid shape of transformation matrix")
        self.params = matrix
        self._coeffs = range(matrix.size - 1)

    @property
    def _inv_matrix(self):
        return np.linalg.inv(self.params)

    def _apply_mat(self, coords, matrix):
        ndim = matrix.shape[0] - 1
        coords = np.array(coords, copy=False, ndmin=2)

        src = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
        dst = src @ matrix.T

        # below, we will divide by the last dimension of the homogeneous
        # coordinate matrix. In order to avoid division by zero,
        # we replace exact zeros in this column with a very small number.
        dst[dst[:, ndim] == 0, ndim] = np.finfo(float).eps
        # rescale to homogeneous coordinates
        dst[:, :ndim] /= dst[:, ndim:ndim+1]

        return dst[:, :ndim]

    def __array__(self, dtype=None):
        if dtype is None:
            return self.params
        else:
            return self.params.astype(dtype)

    def __call__(self, coords):

        return self._apply_mat(coords, self.params)

    @property
    def inverse(self):
        """Return a transform object representing the inverse."""
        return type(self)(matrix=self._inv_matrix)

    def estimate(self, src, dst, weights=None):
        src = np.asarray(src)
        dst = np.asarray(dst)
        n, d = src.shape

        src_matrix, src = _center_and_normalize_points(src)
        dst_matrix, dst = _center_and_normalize_points(dst)
        if not np.all(np.isfinite(src_matrix + dst_matrix)):
            self.params = np.full((d + 1, d + 1), np.nan)
            return False

        # params: a0, a1, a2, b0, b1, b2, c0, c1
        A = np.zeros((n * d, (d+1) ** 2))
        # fill the A matrix with the appropriate block matrices; see docstring
        # for 2D example — this can be generalised to more blocks in the 3D and
        # higher-dimensional cases.
        for ddim in range(d):
            A[ddim*n : (ddim+1)*n, ddim*(d+1) : ddim*(d+1) + d] = src
            A[ddim*n : (ddim+1)*n, ddim*(d+1) + d] = 1
            A[ddim*n : (ddim+1)*n, -d-1:-1] = src
            A[ddim*n : (ddim+1)*n, -1] = -1
            A[ddim*n : (ddim+1)*n, -d-1:] *= -dst[:, ddim:(ddim+1)]

        # Select relevant columns, depending on params
        A = A[:, list(self._coeffs) + [-1]]

        # Get the vectors that correspond to singular values, also applying
        # the weighting if provided
        if weights is None:
            _, _, V = np.linalg.svd(A)
        else:
            weights = np.asarray(weights)
            W = np.diag(np.tile(np.sqrt(weights / np.max(weights)), d))
            _, _, V = np.linalg.svd(W @ A)

        # if the last element of the vector corresponding to the smallest
        # singular value is close to zero, this implies a degenerate case
        # because it is a rank-defective transform, which would map points
        # to a line rather than a plane.
        if np.isclose(V[-1, -1], 0):
            self.params = np.full((d + 1, d + 1), np.nan)
            return False

        H = np.zeros((d+1, d+1))
        # solution is right singular vector that corresponds to smallest
        # singular value
        H.flat[list(self._coeffs) + [-1]] = - V[-1, :-1] / V[-1, -1]
        H[d, d] = 1

        # De-center and de-normalize
        H = np.linalg.inv(dst_matrix) @ H @ src_matrix

        # Small errors can creep in if points are not exact, causing the last
        # element of H to deviate from unity. Correct for that here.
        H /= H[-1, -1]

        self.params = H

        return True

    def __add__(self, other):
        """Combine this transformation with another."""
        if isinstance(other, ProjectiveTransform):
            # combination of the same types result in a transformation of this
            # type again, otherwise use general projective transformation
            if type(self) == type(other):
                tform = self.__class__
            else:
                tform = ProjectiveTransform
            return tform(other.params @ self.params)
        else:
            raise TypeError("Cannot combine transformations of differing "
                            "types.")

    def __nice__(self):
        """common 'paramstr' used by __str__ and __repr__"""
        npstring = np.array2string(self.params, separator=', ')
        paramstr = 'matrix=\n' + textwrap.indent(npstring, '    ')
        return paramstr

    def __repr__(self):
        """Add standard repr formatting around a __nice__ string"""
        paramstr = self.__nice__()
        classname = self.__class__.__name__
        classstr = classname
        return f'<{classstr}({paramstr}) at {hex(id(self))}>'

    def __str__(self):
        """Add standard str formatting around a __nice__ string"""
        paramstr = self.__nice__()
        classname = self.__class__.__name__
        classstr = classname
        return f'<{classstr}({paramstr})>'

    @property
    def dimensionality(self):
        """The dimensionality of the transformation."""
        return self.params.shape[0] - 1

class EuclideanTransform(ProjectiveTransform):

    def __init__(self, matrix=None, rotation=None, translation=None,
                 *, dimensionality=2):
        params_given = rotation is not None or translation is not None

        if params_given and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            matrix = np.asarray(matrix)
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            self.params = matrix
        elif params_given:
            if rotation is None:
                dimensionality = len(translation)
                if dimensionality == 2:
                    rotation = 0
                elif dimensionality == 3:
                    rotation = np.zeros(3)
                else:
                    raise ValueError(
                        'Parameters cannot be specified for dimension '
                        f'{dimensionality} transforms'
                    )
            else:
                if not np.isscalar(rotation) and len(rotation) != 3:
                    raise ValueError(
                        'Parameters cannot be specified for dimension '
                        f'{dimensionality} transforms'
                    )
            if translation is None:
                translation = (0,) * dimensionality

            if dimensionality == 2:
                self.params = np.array([
                    [math.cos(rotation), - math.sin(rotation), 0],
                    [math.sin(rotation),   math.cos(rotation), 0],
                    [                 0,                    0, 1]
                ])
            elif dimensionality == 3:
                self.params = np.eye(dimensionality + 1)
                self.params[:dimensionality, :dimensionality] = (
                    _euler_rotation_matrix(rotation)
                )
            self.params[0:dimensionality, dimensionality] = translation
        else:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)

    def estimate(self, src, dst):

        self.params = _umeyama(src, dst, False)

        # _umeyama will return nan if the problem is not well-conditioned.
        return not np.any(np.isnan(self.params))

    @property
    def rotation(self):
        if self.dimensionality == 2:
            return math.atan2(self.params[1, 0], self.params[1, 1])
        elif self.dimensionality == 3:
            # Returning 3D Euler rotation matrix
            return self.params[:3, :3]
        else:
            raise NotImplementedError(
                'Rotation only implemented for 2D and 3D transforms.'
            )

    @property
    def translation(self):
        return self.params[0:self.dimensionality, self.dimensionality]

class SimilarityTransform(EuclideanTransform):
    def __init__(self, matrix=None, scale=None, rotation=None,
                 translation=None, *, dimensionality=2):
        self.params = None
        params = any(param is not None
                     for param in (scale, rotation, translation))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            matrix = np.asarray(matrix)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            else:
                self.params = matrix
                dimensionality = matrix.shape[0] - 1
        if params:
            if dimensionality not in (2, 3):
                raise ValueError('Parameters only supported for 2D and 3D.')
            matrix = np.eye(dimensionality + 1, dtype=float)
            if scale is None:
                scale = 1
            if rotation is None:
                rotation = 0 if dimensionality == 2 else (0, 0, 0)
            if translation is None:
                translation = (0,) * dimensionality
            if dimensionality == 2:
                ax = (0, 1)
                c, s = np.cos(rotation), np.sin(rotation)
                matrix[ax, ax] = c
                matrix[ax, ax[::-1]] = -s, s
            else:  # 3D rotation
                matrix[:3, :3] = _euler_rotation_matrix(rotation)

            matrix[:dimensionality, :dimensionality] *= scale
            matrix[:dimensionality, dimensionality] = translation
            self.params = matrix
        elif self.params is None:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)

    def estimate(self, src, dst):

        self.params = _umeyama(src, dst, estimate_scale=True)

        # _umeyama will return nan if the problem is not well-conditioned.
        return not np.any(np.isnan(self.params))

    @property
    def scale(self):
        # det = scale**(# of dimensions), therefore scale = det**(1/ndim)
        if self.dimensionality == 2:
            return np.sqrt(np.linalg.det(self.params))
        elif self.dimensionality == 3:
            return np.cbrt(np.linalg.det(self.params))
        else:
            raise NotImplementedError(
                'Scale is only implemented for 2D and 3D.')

class AffineTransform(ProjectiveTransform):

    def __init__(self, matrix=None, scale=None, rotation=None, shear=None,
                 translation=None, *, dimensionality=2):
        params = any(param is not None
                     for param in (scale, rotation, shear, translation))

        # these parameters get overwritten if a higher-D matrix is given
        self._coeffs = range(dimensionality * (dimensionality + 1))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        if params and dimensionality > 2:
            raise ValueError('Parameter input is only supported in 2D.')
        elif matrix is not None:
            matrix = np.asarray(matrix)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            else:
                dimensionality = matrix.shape[0] - 1
                nparam = dimensionality * (dimensionality + 1)
            self._coeffs = range(nparam)
            self.params = matrix
        elif params:  # note: 2D only
            if scale is None:
                scale = (1, 1)
            if rotation is None:
                rotation = 0
            if shear is None:
                shear = 0
            if translation is None:
                translation = (0, 0)

            if np.isscalar(scale):
                sx = sy = scale
            else:
                sx, sy = scale

            if np.isscalar(shear):
                shear_x, shear_y = (shear, 0)
            else:
                shear_x, shear_y = shear

            a0 = sx * (
                math.cos(rotation) + math.tan(shear_y) * math.sin(rotation)
            )
            a1 = -sy * (
                math.tan(shear_x) * math.cos(rotation) + math.sin(rotation)
            )
            a2 = translation[0]

            b0 = sx * (
                math.sin(rotation) - math.tan(shear_y) * math.cos(rotation)
            )
            b1 = -sy * (
                math.tan(shear_x) * math.sin(rotation) - math.cos(rotation)
            )
            b2 = translation[1]
            self.params = np.array([[a0, a1, a2], [b0, b1, b2], [0, 0, 1]])
        else:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)

    @property
    def scale(self):
        if self.dimensionality != 2:
            return np.sqrt(np.sum(self.params ** 2, axis=0))[:self.dimensionality]
        else:
            ss = np.sum(self.params ** 2, axis=0)
            ss[1] = ss[1] / (math.tan(self.shear)**2 + 1)
            return np.sqrt(ss)[:self.dimensionality]

    @property
    def rotation(self):
        if self.dimensionality != 2:
            raise NotImplementedError(
                'The rotation property is only implemented for 2D transforms.'
            )
        return math.atan2(self.params[1, 0], self.params[0, 0])

    @property
    def shear(self):
        if self.dimensionality != 2:
            raise NotImplementedError(
                'The shear property is only implemented for 2D transforms.'
            )
        beta = math.atan2(- self.params[0, 1], self.params[1, 1])
        return beta - self.rotation

    @property
    def translation(self):
        return self.params[0:self.dimensionality, self.dimensionality]


HOMOGRAPHY_TRANSFORMS = (
    SimilarityTransform,
    AffineTransform,
    ProjectiveTransform
)
