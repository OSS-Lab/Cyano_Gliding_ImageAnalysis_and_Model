import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
from numpy.lib.stride_tricks import sliding_window_view
from skimage.morphology import skeletonize
from scipy.signal import find_peaks, savgol_filter, find_peaks_cwt
from bisect import bisect_left
from scipy import integrate
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
import sys
#sys.path.insert(0, 'C:\\Warwick\\Jerko_Code_And_Docs\\FCCP')
sys.path.insert(0, '/Users/lfsmbn/orkun/projects/ongoing/GBMF_motility/Jerko_files/FCCP/')
import Filament_Functions.lowmag_analysis as dio
class SimpleSpline:
    
    def __init__(self, dx, x_axis, spline_fitted_y, derivative_y, spline_object = None):
        """
        Creates SplineLine object from raw (row, col) data, and a desired smoothing degree.
        
        :param points_to_fit: numpy array of shape (N, 2) containing raw data to fit
        :param smoothing_degree: integer, smoothing degree of scipy.interpolate.UnivariateSpline
        """
        self.xy = np.column_stack((x_axis,spline_fitted_y))
        self.dx = dx
        self.dydx = np.column_stack((x_axis,derivative_y(x_axis)))
        self.derivative_func = derivative_y
        self.distance_map = None
        self.within_image_shape = False
        self.spline_object = spline_object

    def crop_to_image_dimensions(self, image_shape, custom_spline = None):
        
        lower_bound, upper_bound = [0,0], image_shape[::-1]
        
        args_within_image = lambda points: np.all((points >= lower_bound) & (points < upper_bound), axis = 1)
        
        if custom_spline is None:
            good_args = args_within_image(self.xy)
            self.xy = self.xy[good_args]
            self.dydx = self.dydx[good_args]
            self.within_image_shape = True
        else:
            return custom_spline[args_within_image(custom_spline)]

    def make_binary_mask(self, binary_image):
        
        if binary_image.dtype != bool:
            raise TypeError('Binary_image needs to be passed as dtype=bool')
        
        if self.within_image_shape is None:
            self.crop_to_image_dimensions(binary_image.shape)
            self.within_image_shape = True
        
        xy_integers = np.floor(self.xy).astype(int)
        within_region = binary_image[xy_integers[:,1], xy_integers[:,0]]

        offset_left = bisect_left(within_region, True)
        
        return within_region, offset_left

    def shift_spline(self, shift_by):
        """Perpendicularly shift every point of the spline by `distance`"""
        rot = np.sign(shift_by)*np.array([[0,-1],[1,0]]) # +90 rot
        all_shifted = []

        for k in range(0,len(self.xy)):
            if k !=0 and k!=len(self.xy)-1:
                tangent = self.xy[k+1] - self.xy[k-1]
            elif k==0:
                tangent = self.xy[1] - self.xy[0]
            else:
                tangent = self.xy[-1]- self.xy[-2]

            normal = np.dot(rot, tangent.reshape(2,1))
            normal /= np.linalg.norm(normal)
            shift = (np.abs(shift_by)*normal).reshape(2,)
            point_shifted = self.xy[k] + shift
            all_shifted.append(point_shifted)
        
        return np.array(all_shifted)
    
    def normal_line_coeffs(self, custom_xy = None, custom_dydx = None) -> tuple[np.ndarray,np.ndarray]:
        """Return coefficients (a,b) of normal y = a*x + b to the spline evaluated at `x_values`. 
        Returns a's and b's """
        if custom_xy is not None and custom_dydx is not None:
            xy = custom_xy
            dydx = custom_dydx
        else:
            xy = self.xy
            dydx = self.dydx
        
        a = (-1 / dydx[:,1])
        b = (xy[:,1] - a * xy[:,0])
        
        return a.reshape(len(xy),1), b.reshape(len(xy),1)
    
    
    def integrate_distance(self,start_x: float, end_x: float) -> float:
        """ Integral to compute arc length of the spline between `start_x` and `end_x` """
        
        arc_len_integrand = lambda x: np.sqrt(1 + self.derivative_func(x)**2) # can have array
 
        if self.distance_map is None:
            distance, _ = quad(arc_len_integrand, start_x, end_x)
        else:
            print('Warning: This is being redone, result possibly wrong.')
            nearest_arg = bisect_left(self.distance_map[:,0], end_x) - 1
            distance = self.distance_map[nearest_arg,1] + \
                        quad(arc_len_integrand, self.distance_map[nearest_arg,0], end_x)[0] 
        return np.abs(distance)


    def x_values_to_distances(self, points_x, reference_point = None):  # this needs a custom spline accomodation
        """
        Computes distances to all x-points in `points_x` from the first point of `reference_point`.

        Args:
            points_x (np.ndarray): An array of x-coordinates of points along the spline.
            reference_point (np.ndarray, optional): Reference point for distance calculation. Default is None

        Returns:
            np.ndarray: An array of distances from the reference point to each point in `points_x`.
        """
        if reference_point is None:
            reference_point = self.xy[0,0] # can I do this?
        
        distances = np.zeros_like(points_x)

        for n, point_x in enumerate(points_x):
            distances[n] = self.integrate_distance(reference_point, point_x)

        return distances
    
    
    @classmethod
    def from_binary_image(cls, dx, input_image, prune_n = 7, spline_k = 4, spline_s = None):
        """Convenience function. Instantiates a SplineLine object by skeletonizing a binary image"""
        
        image_copy = np.copy(input_image)
        # Construct a skeleton
        image_skeleton = skeletonize(image_copy.astype(np.uint8))
        pruned_skeleton = dio.prune_skeleton(image_skeleton.astype(np.uint8), prune_n)

        # Get the (X,Y) points of the pruned skeleton
        pruned_points = np.argwhere(pruned_skeleton[0]>0)

        # Sort indices by X value
        sorted_indices = np.argsort(pruned_points[:,1])
        sorted_points = pruned_points[sorted_indices] + np.array([0.5,0.5]).reshape(1,2)

        spline_fit = UnivariateSpline(sorted_points[:,1], sorted_points[:,0], k=spline_k, s=spline_s)

        x_low = np.min(sorted_points[:,1])
        x_high = np.max(sorted_points[:,1])
        x_axis = np.arange(x_low, x_high, dx) 
        spline_fitted_y = spline_fit(x_axis)
        derivative_y = spline_fit.derivative()
 
        return cls(dx, x_axis, spline_fitted_y, derivative_y, spline_fit)
    
    @classmethod
    def from_csv(cls, axis_path, resample_dx, rot90=False, col_offset = 428):
        axis_points = np.genfromtxt(axis_path, delimiter=',', skip_header=1)
        if rot90==True:
            axis_points = np.column_stack((axis_points[:,1],col_offset-axis_points[:,0]))[::-1]
        
        axis_spline = UnivariateSpline(axis_points[:,0], axis_points[:,1], k=5)
        x_axis = np.arange(axis_points[0,0], axis_points[-1,0], resample_dx)
        return cls(resample_dx, x_axis, axis_spline(x_axis), axis_spline.derivative(), axis_spline)
    
    def _errf(self, t, point_x, point_y):
        """ Calculate the "difference vector" between `point` and a point on the spline
        [t, spline(t)]. If the dot product of the vector and the tangent spline is 0,
        `point` is orthogonal to spline
        """
        difference_vector = np.array([t-point_x, self.spline_object(t) - point_y]).reshape(2,)
        tangent_vector = np.array([1, self.derivative_func(t).item()])
        tangent_vector /= np.linalg.norm(tangent_vector)
        return np.abs(np.dot(difference_vector, tangent_vector))
    
    def transform_coordinates(self, point_x, point_y, distance_from = 'head'):
        """Transform `y` the coordinate of `point` to convert it into orthogonal distance from spline.
         Function also returns the horizontal distance, given the distance_from param. """

        spline_bounds = [(self.xy[0,0],  self.xy[-1,0])]

        optimization_result = minimize(self._errf, x0=point_x, args=(point_x, point_y), bounds=spline_bounds)
        projection_x = optimization_result.x[0]
        projection_y = self.spline_object(projection_x)

        vertical_dist = euclidean([point_x, point_y], [projection_x, projection_y])
        sign = -np.sign(point_y - self.spline_object(projection_x))
        
        if distance_from == 'head':
            horizontal_dist = self.integrate_distance(self.xy[0,0], projection_x)
        elif distance_from == 'tail':
            horizontal_dist = self.integrate_distance(projection_x, self.xy[-1,0])
        else:
            raise Exception('Invalid `distance_from` parameter passed.')
        
        return sign*vertical_dist, np.abs(horizontal_dist)
