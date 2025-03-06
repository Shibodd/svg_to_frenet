SVG_PATH = 'drawing3.svg' # Path to the SVG file. It should have paths with IDs 'outer', 'inner', 'raceline'.

CLOCKWISE = True # Whether the resulting path should go clockwise or counterclockwise
START_OFFSET_S = 0.1 # [0-1] position of starting line in the SVG 'raceline' spline. Not linear. [adim]ss

RACELINE_POINTS_PER_METER = 20 # The resolution of the final result [pt/m]
METERS_PER_PIXEL = 0.05 # Map resolution [m/px]

SPLINE_CONVERSION_N_SAMPLES_MUL = 2 # Multiplies RACELINE_POINTS_PER_METER when sampling the SVG path to create the raceline spline [adim]
SPLINE_CONVERSION_SMOOTHING = 10 # Smoothing factor for the raceline spline. Minimum 0 (no smoothing) [??]

RAY_LENGTH_METERS = 100 # Raycasting length for finding boundaries (aka, maximum centerline distance from boundary) [m]

import svgpathtools
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.integrate
import math

def get_paths(filename) -> tuple[svgpathtools.Path, svgpathtools.Path, svgpathtools.Path]:
    paths, attributes = svgpathtools.svg2paths(filename)

    assert(len(paths) == len(attributes))

    paths_by_name = {}
    for path, attr in zip(paths, attributes):
        paths_by_name[attr.get('id')] = path
    
    return paths_by_name['raceline'], paths_by_name['outer'], paths_by_name['inner']

def c2v(c):
    return np.array([c.real, c.imag])
def v2c(v):
    return complex(v[0], v[1])

def sample(fcn, N):
    ans = np.zeros((2, N))
    for i in range(N):
        s = i / (N - 1)
        ans[:,i] = c2v(fcn(s))
    return ans

def points_to_spline(pts, invert=False, offset=0):
    if invert:
        pts = np.flip(pts, 1)
    if offset:
        pts = np.roll(pts, offset, 1)

    tck, _ = scipy.interpolate.splprep(pts, s=SPLINE_CONVERSION_SMOOTHING, per=1)
    return tck

def curvature(dxs, ddxs):
    return (dxs[0,:] * ddxs[1,:] - dxs[1,:] * ddxs[0,:]) / (dxs[0,:]**2 + dxs[1,:]**2) ** 1.5

def dlength(s, tck):
    # The derivative of the length is just the 2-norm of the derivative
    dx, dy = scipy.interpolate.splev(s, tck, 1)
    return np.sqrt(np.square(dx) + np.square(dy))

def get_closest_intersection(ray_length, intersections):
    # [((T1, seg1, t1), (T2, seg2, t2)), ...]

    if len(intersections) <= 0:
        return None
    
    # Line param for intersection
    intersection_params = np.array([ x[1][2] for x in intersections ])

    intersection_idx = np.argmin(intersection_params)
    return abs(intersection_params[intersection_idx]) * ray_length

def wrap(value, min, max):
    w = max - min
    tmp = math.fmod(value, w)
    if (tmp > max):
        tmp -= w
    elif tmp < min:
        tmp += w
    return tmp

# Read data
raceline, outer, inner = get_paths(SVG_PATH)

# Sample the raceline
raceline_spline_n = math.ceil(raceline.length() * METERS_PER_PIXEL * RACELINE_POINTS_PER_METER * SPLINE_CONVERSION_N_SAMPLES_MUL)
original_raceline_pts = sample(lambda s: raceline.point(wrap(s + START_OFFSET_S, 0, 1)), raceline_spline_n) * METERS_PER_PIXEL

# Create a spline
raceline_tck = points_to_spline(original_raceline_pts)

# Compute raceline length
raceline_len_meters = scipy.integrate.quad(dlength, 0, 1, args=(raceline_tck,), epsabs=2/RACELINE_POINTS_PER_METER)[0]
print("Raceline is", raceline_len_meters, "meters")

# Compute mean curvature
mean_k = scipy.integrate.quad(lambda s: curvature(
    np.vstack(scipy.interpolate.splev(s, raceline_tck, der=1)),
    np.vstack(scipy.interpolate.splev(s, raceline_tck, der=2)),
), 0, 1, epsabs=1/raceline_len_meters)[0]

assert(abs(mean_k) >= 1/raceline_len_meters) # Otherwise we'd not be sure of the curvature sign.

raceline_is_clockwise = mean_k < 0
print(f'Input raceline is {"clockwise" if raceline_is_clockwise else "counterclockwise"} (mean k: {mean_k})')

# Create a spline going in the correct direction
raceline_tck = points_to_spline(original_raceline_pts, raceline_is_clockwise != CLOCKWISE)

# Compute how many samples we should take, and the sample points
n = math.ceil(raceline_len_meters * RACELINE_POINTS_PER_METER)
print("Sampling with", n, "points")

# Sample the raceline geometry.
spline_ss = np.linspace(0, 1, n)
ss = spline_ss * raceline_len_meters
xs = np.vstack(scipy.interpolate.splev(spline_ss, raceline_tck, 0))
dxs = np.vstack(scipy.interpolate.splev(spline_ss, raceline_tck, 1))
ddxs = np.vstack(scipy.interpolate.splev(spline_ss, raceline_tck, 2))

# Compute the curvature.
ks = curvature(dxs, ddxs)

# Compute tangents and normals
ts = dxs / np.linalg.norm(dxs, axis=0)
ns = np.vstack((-ts[1,:], ts[0,:]))

quiver_n = min(xs.shape[1], math.ceil(raceline_len_meters))
quiver_step = math.floor(xs.shape[1] / quiver_n)

starting_line_pts = np.vstack([xs[:,0] - ns[:,0], xs[:,0] + ns[:,0]]).T

plt.plot(original_raceline_pts[0,:], original_raceline_pts[1,:], label='Original raceline')
plt.plot(starting_line_pts[0,:], starting_line_pts[1,:], label='Starting line')
plt.plot(xs[0,:], xs[1,:], label='Final raceline')
plt.quiver(xs[0,::quiver_step], xs[1,::quiver_step], ts[0,::quiver_step], ts[1,::quiver_step], scale=30, width=0.005, headwidth=1, headlength=0.5, angles='xy')
plt.quiver(xs[0,::quiver_step], xs[1,::quiver_step], ns[0,::quiver_step], ns[1,::quiver_step], scale=30, width=0.005, headwidth=1, headlength=0.5, angles='xy')
plt.axis('equal')
plt.legend()

# Get the left and right bounds
left_bound, right_bound = (outer, inner) if CLOCKWISE else (inner, outer)

# Compute n_left and n_right
n_lefts = np.zeros(n)
n_rights = np.zeros(n)

# Just use a raycasting approach
ray_length_pixels = RAY_LENGTH_METERS / METERS_PER_PIXEL
for i in range(n):
    # Convert back to pixels, as we're just using the svg library functionality for this
    x = v2c(xs[:,i]) / METERS_PER_PIXEL
    n = v2c(ns[:,i])

    # Shoot a ray to the left and right of the current point, and register an hit with the corresponding bound
    n_left_px = get_closest_intersection(ray_length_pixels, left_bound.intersect(svgpathtools.Line(x, x + n * ray_length_pixels))) or ray_length_pixels
    n_right_px = get_closest_intersection(ray_length_pixels, right_bound.intersect(svgpathtools.Line(x, x - n * ray_length_pixels))) or ray_length_pixels

    # Convert back to meters and store
    n_lefts[i] = n_left_px * METERS_PER_PIXEL
    n_rights[i] = n_right_px * METERS_PER_PIXEL

# Plot the boundaries for debugging
left_projection = xs + ns * n_lefts
right_projection = xs - ns * n_rights

original_left_bound_pts = sample(lambda s: left_bound.point(wrap(s, 0, 1)), raceline_spline_n) * METERS_PER_PIXEL
original_right_bound_pts = sample(lambda s: right_bound.point(wrap(s, 0, 1)), raceline_spline_n) * METERS_PER_PIXEL

plt.figure()
plt.plot(original_raceline_pts[0,:], original_raceline_pts[1,:], linewidth=0.5, label='Original raceline')
plt.plot(original_left_bound_pts[0,:], original_left_bound_pts[1,:], linewidth=0.5, label='Original left bound')
plt.plot(original_right_bound_pts[0,:], original_right_bound_pts[1,:], linewidth=0.5, label='Original right bound')
plt.scatter(xs[0,:], xs[1,:], s=0.8, label='Final raceline')
plt.scatter(left_projection[0,:], left_projection[1,:], s=0.8, label='n_left projection')
plt.scatter(right_projection[0,:], right_projection[1,:], s=0.8, label='n_right projection')
plt.axis('equal')
plt.legend()

# Also plot curvature, n_left, n_right
fig, axs = plt.subplots(3, sharex=True)
fig.supxlabel('s [m]')

axs[0].plot(ss, n_lefts)
axs[0].set_ylabel('n_left [m]')

axs[1].plot(ss, n_rights)
axs[1].set_ylabel('n_right [m]')

axs[2].plot(ss, ks)
axs[2].set_ylabel('k [1/m]')
plt.show()