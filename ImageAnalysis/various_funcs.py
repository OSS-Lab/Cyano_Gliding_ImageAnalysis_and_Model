import sys
import numpy as np
#sys.path.insert(0, 'C:\\Warwick\\Jerko_Code_And_Docs\\FCCP')
sys.path.insert(0, r'/Users/lfsmbn/orkun/projects/ongoing/GBMF_motility/Jerko_files/Manuscript_Code/')
#import Filament_Functions.lowmag_analysis as dio
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy import interpolate
from skimage.measure import label, regionprops
from scipy.signal import convolve2d
from scipy import ndimage as ndi
from skimage.filters import threshold_niblack,hessian
from skimage.morphology import disk, opening
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import linregress


def estimate_peaks_hessian(image, masking_image, filament_axis):
    hess_profile = filament_axis.laterally_integrate(hessian(image), masking_image)
    hessian_thresholded = hess_profile[:,0] < 0.3*np.median(hess_profile[:,0])
    
    redundant_args = np.flatnonzero(np.diff(hessian_thresholded) == 1)
    hessian_thresholded[redundant_args] = False
    
    hess_peaks = find_peaks(hessian_thresholded.astype(int))[0]

    return hess_profile[:,1][hess_peaks]

def niblack_threshold(frame, window_size = 25, k = 0.27 , selem = disk(5), blob_size_thr = 1000):
    binary_niblack = frame > threshold_niblack(frame, window_size=window_size, k=k)
    opened_niblack = opening(binary_niblack,selem)
    opened_niblack = filter_threshold(opened_niblack, 0.8, blob_size_thr).astype(np.int8)
    return opened_niblack

def filament_spline(img, prune_n = 7, spline_k = 4):

    # Construct a skeleton
    kernels = dio.load_kernels()
    skl = skeletonize(img.astype(np.uint8), method = 'lee')
    pruned = dio.prune_skeleton(skl, kernels, prune_n)

    # Get the (X,Y) points of the pruned skeleton
    data = np.argwhere(pruned[0]>0)

    # Split X and Y
    spl_x = data[:,1]
    spl_y = data[:,0]

    # Sort indices by X value
    sorted_ind = np.argsort(spl_x)

    # Apply the sorted index
    spl_x = spl_x[sorted_ind]
    spl_y = spl_y[sorted_ind]

    # Fit spline to this skeleton
    spline = interpolate.UnivariateSpline(spl_x,spl_y, k = spline_k)
    x_interp = np.arange(0, np.max(spl_x), 0.01)
    y_interp = spline(x_interp)
    
    spl_twocol = np.array([x_interp, y_interp]).T
    
    return spl_twocol

def simple_threshold(image, thr_val):
    binary = image > thr_val
    return binary.astype(np.uint8)
    
def filter_threshold(image, thr_val, min_blob_area):
    binary = image > thr_val
    binary =  binary.astype(np.uint8)
    labelled_binary = label(binary)
    props = regionprops(labelled_binary)
    to_remove =[p.label for p in props if p.area<min_blob_area]
    for item in to_remove:
        binary[labelled_binary == item] = 0
    return binary
    
    
def exclude(prev_filament, image):
    to_exclude = np.argwhere(prev_filament > 0)
    excluded = np.copy(image)
    for point in to_exclude:
        excluded[point[0], point[1]] = 0
    return excluded

def spline_erase(spline, image, mode, cursor=['left',0]):
    image_copy = np.copy(image)
    spline_copy = np.copy(spline).astype(int)
    uniques = []

    # Pixelate the spline
    for k in range(0, len(spline_copy)):
        point  =  spline_copy[k].tolist()
        if point not in uniques:
            # make sure the spline fits within the image
            if point[1] < image.shape[0] and point[1] > 0:
                if point[0] < image.shape[1] and point[0] > 0:
                    uniques.append(point)
            
    for point in uniques:
        if mode == 'under':  
            to_del = range(point[1],image_copy.shape[0])                       
        elif mode == 'over': 
            to_del = range(point[1], -1, -1)
        elif mode == 'left':
            to_del = range(point[0], -1, -1)
        elif mode == 'right':
            to_del = range(point[0], image_copy.shape[1])
        
        
        for n in to_del:
            if mode == 'under' or mode == 'over':
                image_copy[n, point[0]] = 0
            if mode == 'left' or mode == 'right':
                image_copy[point[1], n] = 0
    
    # why did I do this? 
    if cursor[0] == 'left':
        if cursor[1] != 0:
            n = 0
            while n<=cursor[1]:
                image_copy[:,n] = 0
                n+=1
    
    if cursor[0] == 'right':
        if cursor[1] != 0:
            n = cursor[1]
            while n<image.shape[1]:
                image_copy[:,n] = 0
                n+=1

    return image_copy

def perpendicular(vector, distance, orientation):
    rot = orientation*np.array([[0,-1],[1,0]]) # +90 rot
    
    normal = np.dot(rot, vector.reshape(2,1))
    normal = normal/np.linalg.norm(normal)
    
    return (distance*normal).reshape(2,)

# Manual tracking stuff
def intensity_profile(image, filament_axis, spline_k = 3, spline_s = 0.4):
    profile = filament_axis.pixel_intensity_profile(image, normalized=True)
    profile_spline = UnivariateSpline(profile[0], profile[1], k=spline_k, s=spline_s)
    resampled_x = np.arange(profile[0][0], profile[0][-1], 0.01)
    profile_at_x = profile_spline(resampled_x)
    return resampled_x, profile_at_x

def search_area_mask(prev_peak_loc, x_axis, search_radius  = 1):
    column_x = x_axis[:, None] # Cast into column vector
    mask_low = column_x > (prev_peak_loc - search_radius)
    mask_high = column_x < (prev_peak_loc + search_radius)
    return np.any(mask_low & mask_high, axis = 1)
    
def minima_masked_profile(profile, mask):
    labelled_mask, num_components = ndi.label(mask)
    min_values = ndi.minimum_position(profile, labels=labelled_mask, index=np.arange(1, num_components + 1))
    return [item[0] for item in min_values]

def fix_septae(profile, x_axis, locations):
    peak_locations = []
    for location in locations:
        area_mask = search_area_mask(location, x_axis, search_radius=3)
        peak = minima_masked_profile(profile, area_mask)[0]
        peak_locations.append(peak)
    return peak_locations


def calculate_angles(group, min_radius = 3, sample_radius = 2, group_len_thr = 10):
    """  """
    
    if group['height'].max() < min_radius:
        group['angle'] = None
        return group
    if group['height'].min() > -min_radius:
        group['angle'] = None
        return group
    
    sorted_group = group.sort_values('distance')
    sorted_group = sorted_group[(sorted_group['height'].abs() < sample_radius)]
    
    if len(sorted_group) < group_len_thr:
        group['angle'] = None
        return group

    slope, *_ = linregress(sorted_group['distance'], sorted_group['height'])
    angle = np.arctan(slope)

    group['angle'] = np.rad2deg(angle)

    return group

def assign_to_cell(group, details_df, max_radius_px = 10, min_radius_px = 0.25):
    
    all_septae_present = details_df['particle'].unique().tolist()
    # Select columns from the input data
    tracks_df = group[['frame', 'distance']].astype({'frame': np.int16})

    # Match all the septae from details_df which are present in frames corresponding to the track TRACK_ID
    merged_df = pd.merge(tracks_df, details_df, how='left', left_on='frame', right_on='frame')
    
    merged_df['delta'] = merged_df['distance'] - merged_df['distance']
    merged_df = merged_df[merged_df['delta'].abs() < max_radius_px]
    unique_septae_in_merged = merged_df['particle'].unique()

    if len(unique_septae_in_merged) !=1:
        group['cell'] = None
        return group
    
    probable_septum = merged_df['particle'].iloc[0] 

    if merged_df['delta'].mean() > min_radius_px:
        cell_assignment = probable_septum
        group['cell'] = cell_assignment

    elif merged_df['delta'].mean() < -min_radius_px:
        if (probable_septum - 1) in all_septae_present:
            cell_assignment = probable_septum - 1
            group['cell'] = cell_assignment
    else:
        group['cell'] = None
    
    return group


def calc_speeds(group, min_size = 10, window_length = 5):
    """Assigns speed sign to each row"""
    # Assign Speeds
    group_ = group.copy()
    if len(group_) < min_size:
        group_['speed'] = None
        group_['speed2'] = None
        return group_
    group_ = group[::3]
    group_ = group_.sort_values('frame')
    #group['smooth_dist'] = np.convolve(group['distance'], smoothing_kernel, mode = 'same')
    group_['smooth_dist'] = savgol_filter(group_['distance'], window_length=window_length, polyorder=2)
    group_['speed'] = group_['smooth_dist'].diff() / group_['frame'].diff()
    group_['speed2'] = group_['distance'].diff() / group_['frame'].diff()

    return group_


def ax_plot_septae(ax, septae, particles_col='particle', time_col='frame', pos_col='x', sorted = False, tracks=None, params_dict={}):
    """
    Plots all the particle tracks from the dataframe `septae`. If the particle ID column is not 'particle', we
    need to pass `particles_col` as an argument. Same if `time_col` != 'frame' and `pos_col` != 'x'.
    
    Optionally, can pass a blob tracks dataframe. This dataframe needs to have its columns be in order of
    ['TRACK_ID', 'TIME_COLUMN', 'POSITION_COLUMN']. 'TRACK_ID' needs to be called that, others can be called whatever.

    params_dict : Dictionary containing optional parameters for customizing the plot:
    - 'v_lines': List of x-coordinates to draw vertical guidelines.
    - 'h_lines': List of y-coordinates to draw horizontal guidelines.
    - 'y_label': Custom label for the y-axis.
    - 'x_label': Custom label for the x-axis.
    - 'track_color': Color for the tracks (if provided).
    """
    
    unique_particles = septae[particles_col].unique()
    colors = plt.cm.inferno(np.linspace(0.3, 1, len(unique_particles) + 1))

    for n, x in enumerate(unique_particles):
        time_axis = septae.loc[septae[particles_col] == x][time_col].values
        pos_axis = septae.loc[septae[particles_col] == x][pos_col].values
        if sorted == True:
            ax.plot(time_axis, pos_axis, '-.', markersize=0.8, color=colors[x], label=f'Particle {x}')
        else:
            ax.plot(time_axis, pos_axis, '-.', markersize=0.8, color=colors[n], label=f'Particle {x}')
        ax.annotate(f'{x}', xy=(time_axis[0], pos_axis[0]), xytext=(1, 1), textcoords='offset points')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3)

    params = params_dict.keys()

    if 'v_lines' in params:
        x_coords = params_dict['v_lines']
        _ = [ax.axvline(x, linestyle='--', linewidth=0.8) for x in x_coords]

    if 'h_lines' in params:
        y_coords = params_dict['h_lines']
        _ = [ax.axhline(y, linestyle='--', linewidth=0.8) for y in y_coords]

    if 'y_label' in params:
        ax.set_ylabel(f"{params_dict['y_label']}")
    else:
        ax.set_ylabel('Distance Component [px]')

    if 'x_label' in params:
        ax.set_xlabel(f"{params_dict['x_label']}")
    else:
        ax.set_xlabel('Time [Frames]')

    if not isinstance(tracks, pd.DataFrame):
        return ax

    if 'track_color' in params:
        track_color = params_dict['track_color']
    else:
        track_color = 'pink'

    for name, group in tracks.groupby('TRACK_ID'):
        time = group.iloc[:, 1]
        distance = group.iloc[:, 2]
        ax.plot(time, distance, '.', markersize=0.8, color=track_color, label=f'Track {name}')
        ax.annotate(f'{name}', xy=(time[-1], distance[-1]), xytext=(1, 1), textcoords='offset points', color=track_color)

    return ax

def summarize_mean_distances(group, frame_range):
    return group.loc[group['frame'].between(frame_range[0], frame_range[1]), 'distance'].mean()

def sort_particles(dataframe, frame_range):
    """
    """
    mean_distances = dataframe.groupby('particle').apply(summarize_mean_distances, frame_range).reset_index()
    mean_distances.columns = ['particle', 'mean_distance']
    # Make a new column by ranking the values, the '-1' ensures the ladder will start from 0
    mean_distances['particle_sorted'] = mean_distances['mean_distance'].rank(method='min').astype(int) - 1
    # Use this to create a mapping dictionary
    mapping_dict = {o: s for o, s in zip(mean_distances['particle'].values, mean_distances['particle_sorted'].values)}
    # New column with correctly sorted particle labels
    dataframe['particle_sorted'] = dataframe['particle'].map(mapping_dict)
    dataframe.drop('particle', inplace = True, axis=1)
    dataframe.rename(columns={'particle_sorted': 'particle'}, inplace=True)
    dataframe.index.names = ['']
    dataframe.reset_index(inplace=True)
    dataframe.drop(columns=[''], inplace=True)
    return dataframe


def ax_septae_on_image(ax, septae, image, fil_axis, marker_size = 1, font_size = 10, frame_no = None):
    if frame_no is None:
        if hasattr(image, 'frame_no'):
            frame_no = image.frame_no
        else:
            raise ValueError('Image/Frame has no attribute `frame_no` in its metadata. Pass `frame_no` to the function manually')
    
    frame_septae = septae.loc[septae['frame'] == frame_no, 'particle'].unique()
    colors = plt.cm.inferno(np.linspace(0.3,1,max(frame_septae)+1))
    for particle in frame_septae:
        x_val = septae.loc[(septae['particle'] == particle) & (septae['frame'] == frame_no), ['x']].values[0][0]
        y_val = fil_axis.spline_object(x_val)
        ax.plot(x_val, y_val, 'x', color = colors[particle], markersize =marker_size)
        ax.annotate(f'{particle}', xy=(x_val, y_val), xytext=(1, 1), textcoords='offset points', color='pink', fontsize=font_size)
    return ax

# this is not needed?
def make_septae_pairs(dataframe, good_pairs):
    pairs_in_frames = {}
    frames = pd.unique(dataframe['frame'])
    for frame in frames:
        sub_df = dataframe.loc[dataframe['frame'] == frame]
        ordered_septae = pd.unique(sub_df['particle'])
        possible_pairs = [[ordered_septae[n-1],ordered_septae[n]] for n in range(1, len(ordered_septae))]
        possible_pairs = [pair for pair in possible_pairs if pair in good_pairs]
        pairs_in_frames[frame] = possible_pairs
    return pairs_in_frames


#def assign_full_cells(group, good_septae):
#    group['full_cell'] = group['particle'].apply(lambda x: True if x in good_septae else False)
#    return group


def assign_full_cells(group, max_septum):
    # input is grouped by frame
    # Some frames may have no septae
    
    if len(group) == 0:
        group['full_cell'] = None
        return group
    
    septae = group['particle'].unique() # septae present
    septae = np.sort(septae)
    good_starting = []
    
    for k in range(len(septae) - 1):
        if septae[k] in [0, max_septum]:
            good_starting.append(septae[k])
        else:
            if (septae[k+1] - septae[k]) == 1:
                good_starting.append(septae[k])
    
    if len(good_starting) > 0:
        group['full_cell'] = group['particle'].apply(lambda x: True if x in good_starting else False)
        return group
    else:
        group['full_cell'] = None
        return group