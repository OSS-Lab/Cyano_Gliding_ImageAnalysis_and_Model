import numpy as np
import cv2
from skimage.morphology import skeletonize, closing, disk
from skimage import feature as sk_feature
import os
import skimage.filters as sk_fil
import skimage.measure as sk_mes
import matplotlib.pyplot as plt
import re
from scipy.ndimage import convolve
from scipy import interpolate
from scipy.stats import linregress
from scipy.spatial.distance import euclidean
import math

kernel_paths = [r'/Documents/Cyano_Gliding_ImageAnalysis_and_Model-main/figures_figureCodes/edge_kernels.npy',
                r'/Documents/Cyano_Gliding_ImageAnalysis_and_Model-main/figures_figureCodes/junction_kernels2.npy',
                r'/Documents/Cyano_Gliding_ImageAnalysis_and_Model-main/figures_figureCodes/end_kernels.npy']
def load_kernels():
    """ Load morphology kernels from files.
    Expects a list of paths of form
    [path_edge, path_junction, path_end]"""

    kernels = []
    for n in range(0,3):
        with open(kernel_paths[n], 'rb') as f:
            kernels.append(np.load(f))
    return kernels
    
def ends_junctions(image, kernels):
    # Find all junctions
    junctions = np.zeros(image.shape).astype('uint8')
    for kernel in kernels[1]:
        junctions +=  cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    # Find all line ends
    end_points = np.zeros(image.shape).astype('uint8')
    for kernel in kernels[2]:
        end_points +=  cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    return end_points, junctions
    
def nearest(image, point):

    very_near = np.array([[1,0],[-1,0],[0,1],[0,-1]])
    near = [[1,1],[-1,-1],[-1,1],[1,-1]]

    # Check for simple neighbors
    very_near_found = [ ind.tolist() for ind in very_near if image[(point + ind)[0],(point + ind)[1]] > 0 ]

    # Clean up 2nd neighbors from 'near' conditional on first neighbor locations
    if len(very_near_found)>0:
        for pt in very_near_found:
            if pt == [0,-1]:
                if [-1,-1] in near:
                    near.remove([-1,-1])
                if ([1,-1]) in near:
                    near.remove([1,-1])
            elif pt == [0,1]:
                if [-1,1] in near:
                    near.remove([-1,1])
                if [1,1] in near:
                    near.remove([1,1])
            elif pt == [-1,0]:
                if [-1,-1] in near:
                    near.remove([-1,-1])
                if [-1,1] in near:
                    near.remove([-1,1])
            elif pt == [1,0]:
                if [1,-1] in near:
                    near.remove([1,-1])
                if [1,1] in near:
                    near.remove([1,1])
    near = np.array(near)

    if len(near) > 0:
        near_found = [ ind.tolist() for ind in near if image[(point + ind)[0],(point + ind)[1]]  > 0 ]
        return point + np.array(near_found + very_near_found)
    else:
        return point + np.array(very_near_found)
        
        
def advance(prev_pt, curr_pt, image):
    if type(prev_pt)==type(0) and prev_pt==0:
        # I expect to find only one
        point = nearest(image,curr_pt)
        if len(point)==1 :
            return point[0]
        else:
            print('First point of skeleton has too many neighbors (>1)')

    elif type(prev_pt) == type(np.array([])):
        points = nearest(image,curr_pt)
        # Now we need to remove the previous point
        points = points.tolist()
        points.remove(prev_pt.tolist())
        return np.array(points[0])

def labelled_image(image, min_area = 100):
    image_copy = np.copy(image)
    denoised = cv2.fastNlMeansDenoising(image,templateWindowSize=5, searchWindowSize=14, h=0.7)

    kernel = np.ones((3,3), np.uint8)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE,kernel, iterations = 1)

    #  Initial yen threshold
    thresh = sk_fil.threshold_yen(denoised)  
    binary = denoised > (0.97*thresh)

    # Label imaghe regions
    labelled = sk_mes.label(binary)
    props = sk_mes.regionprops(labelled)

    candidates =[[p.area, p.label] for p in props if p.area>min_area]
    
    return labelled, candidates

def get_skeletons(candidate, labelled, num_x_pts = 500, n_pts = 5):
    # Iterate over candidate regions, extract skeletons
    results = []

    partial_labelled = (labelled == candidate[1]).astype(np.uint8)
    filtered = cv2.GaussianBlur(partial_labelled, (3, 3), 0)
    skl = skeletonize(partial_labelled, method='lee')
    pruned, junction_flag = prune_skeleton(skl, n_pts) #prune forks

    if junction_flag==0:
        # order the skeleton
        points = np.argwhere(pruned > 0)
        sorted_ind = np.argsort(points[:,1])
        ordered_ = points[sorted_ind]

        # This needs binning
        x_values = np.unique(ordered_[:,1])
        y_values =[]

        for value in x_values:
            args = np.argwhere(ordered_[:,1] == value)
            y_vals = ordered_[:,0][args]
            y_values.append(np.mean(y_vals))

        res= interpolate.UnivariateSpline(x_values,y_values, k=4)
        smooth_x = np.linspace(np.min(x_values),np.max(x_values),num_x_pts)
        smooth_y = res(smooth_x)
        smooth = np.array([smooth_x, smooth_y]).T
    else:
        return None, junction_flag
        
    return smooth, junction_flag
    

def prune_skeleton(image, n_pts):
    kernels=load_kernels()
    # First pad the image
    pruned = np.pad(image,2, constant_values=(0))

    # Remove superfluous corners from skeleton
    for kernel in kernels[0]:
        pruned = pruned  - cv2.morphologyEx(pruned, cv2.MORPH_HITMISS, kernel)

    # Extract line ends and tri-jucntions
    ends, junctions = ends_junctions(pruned, kernels)

    # Images of ends, juncts are returned, I want lists of coordinates here
    ends = np.argwhere(ends>0).tolist()
    junctions = np.argwhere(junctions>0).tolist()

    # Iterate Over Ends

    for end in ends:
        registered_pts = []
        curr_pt = np.array(end)
        prev_pt = 0
        #Step
        n=1
        while n<=n_pts:
            if curr_pt.tolist() not in junctions:
                #print('Current Point', curr_pt)
                registered_pts.append(curr_pt.tolist())
            else:
                break
            # New Assignmments
            next_pt = advance(prev_pt, curr_pt, pruned)
            prev_pt = curr_pt
            curr_pt = next_pt
            n=n+1

        # We want to have hit a junction within n_pts,  if we did, we delete the pts
        if len(registered_pts) < n_pts:
            for pt in registered_pts:
                pruned[pt[0], pt[1]] = 0

    # Fix loops
    junction_flag = 0
    _ , junctions =  ends_junctions(pruned, kernels)

    if len(np.argwhere(junctions > 0)) > 0 :
        # There may be some loops left
        filled_holes = cv2.morphologyEx(pruned, cv2.MORPH_CLOSE, np.ones((3,3)))
        pruned = skeletonize(filled_holes, method = 'lee')
        # Verify that there are no more junctions
        _ , junctions =  ends_junctions(pruned, kernels)

        if len(np.argwhere(junctions > 0)) > 0:
            junction_flag = 1
            return None, junction_flag
        else:
            return pruned[2:-2,2:-2], junction_flag
    else:
        return pruned[2:-2,2:-2], junction_flag
    return pruned[2:-2,2:-2], junction_flag
    


def extend_skeleton(smooth, image, labelled, candidate):
    # Sample beginning and end of spline
    first_10 = smooth[0:10,:]
    last_10 = smooth[-10:,:]

    # delta_x of the spline
    step = np.abs(np.round(smooth[1][0] - smooth[0][0],4))

    #
    ext_1 = fit_linear(first_10, 10, step, mode = 'first')
    ext_2 = fit_linear(last_10, 10, step, mode = 'last')

    ext_smooth = np.concatenate((ext_1, smooth, ext_2))
    
    # Check that ext_smooth fits within the image
    indices_to_remove = []
    y_max, x_max = image.shape
    
    for p,point in enumerate(ext_smooth):
        condition_1 = point[0] >0 and point[1]>0
        condition_2 = point[0] < x_max and point[1] < y_max
        if condition_1 and condition_2:
            pass
        else:
            indices_to_remove.append(p)
            
    ext_smooth = np.delete(ext_smooth, indices_to_remove, axis = 0)
    #

    # now comes the line profile bit
    prof,_ = line_profile(ext_smooth, image) 
    prof_spl_x = np.arange(0, len(ext_smooth),1)

    profile_spline= interpolate.UnivariateSpline(prof_spl_x, prof, k=4)

    prof_spl_y = profile_spline(prof_spl_x)
    cov_spl_y = np.convolve(prof_spl_y, [1,1,1,1,1], mode='same')/5

    avg_noise, std_dev = avg_background(image, labelled, candidate)
    perc_90 = np.percentile(cov_spl_y, 90)
    delta_signal = perc_90 - (avg_noise+2*std_dev)
    profile_thr = 0.5*delta_signal + (avg_noise+2*std_dev)
    thresholded = cov_spl_y > profile_thr
    
    good_args = np.argwhere(thresholded == True)
    good_args = good_args.T[0]
    fixed = ext_smooth[good_args]

    return fixed

def fit_linear(points, n, step, mode):

    if mode == 'first':
        x = np.arange(points[:,0][0] - n, points[:,0][0],  step)
    elif mode == 'last':
        x = np.arange(points[:,0][-1], points[:,0][-1] + n, step)

    a, b = linregress(points[:,0], points[:,1] )[0:2]
    
    y = a*x + b
    
    ext = np.array([x,y]).T
    
    return ext
    

def line_profile(spline, image):
    result=[]
    pts =[]
    for point in spline:
        x = math.floor(point[0])
        y = math.floor(point[1])
        intensity_value = image[y,x]
        result.append(intensity_value)
        pts.append([x,y])
    return result,pts


def avg_background(image, labelled, candidate):
    image_copy = np.copy(image)
    partial_labelled = (labelled == candidate[1]).astype(np.uint8)
    thinner = convolve(partial_labelled, disk(10)) 
    thinner = (thinner>0).astype(np.uint8)
    thicker = convolve(partial_labelled, disk(20)) 
    thicker = (thicker>0).astype(np.uint8)
    shell = thicker - thinner
    partial_mask = shell > 0
    image_copy[~partial_mask] = 0
    avg = np.mean(image_copy[partial_mask])
    stddev = np.std(image_copy[partial_mask])
    
    return avg, stddev
    
    
def mpts_spline(mpts, spline_factor = 1000, k=5):
    sort_ind = np.argsort(mpts[:,0])
    x_values = mpts[:,0][sort_ind]
    y_values = mpts[:,1][sort_ind]
    
    # This needs binning
    x_uniques = np.unique(x_values)
    y_resulting =[]
    
    for value in x_uniques:
        args = np.argwhere(x_values == value)
        y_all = y_values[args]
        y_resulting.append(np.mean(y_all))
    
    res= interpolate.UnivariateSpline(x_values,y_values, k=k)
    smooth_x = np.linspace(np.min(x_values),np.max(x_values),spline_factor)
    smooth_y = res(smooth_x)
    smooth = np.array([smooth_x, smooth_y]).T
    return smooth
    
def worm_length(ordered):
    indices = np.arange(0, len(ordered))
    length = 0
    for k in range(1, len(indices)):
        point_0 = ordered[indices[k-1]]
        point_1 = ordered[indices[k]]
        delta = np.linalg.norm(point_1 -point_0)
        length = length + delta
    return length
    

def mid_point(ordered):
    downsampled_indices = np.arange(0, len(ordered))
    if downsampled_indices[-1] != len(ordered) - 1:
        downsampled_indices[-1] = len(ordered) - 1
    dists = [0]
    length = 0
    for k in range(1, len(downsampled_indices)):
        point_0 = ordered[downsampled_indices[k-1]]
        point_1 = ordered[downsampled_indices[k]]
        delta = np.linalg.norm(point_1 -point_0)
        length = length + delta
        dists.append(length)
    dists = np.array(dists)
    mid_ind = np.argmin(np.abs(dists - length/2))
    return ordered[mid_ind]


def project_to_spline(point, spline, return_index = 1):
    spline_x, spline_y = spline[:,0], spline[:,1]
    p_x, p_y = point[0], point[1]
    nearest_ind = np.argmin(np.abs(spline_x - p_x))
    search_radius_low, search_radius_high = int(0.05*len(spline_x)), int(0.05*len(spline_x))
    if nearest_ind - search_radius_low < 0:
        search_radius_low = nearest_ind
    if nearest_ind + search_radius_high > len(spline_x) - 1:
        search_radius_high = 0
    xy_candidates = spline[nearest_ind - search_radius_low : nearest_ind + search_radius_high,:]
    distances = [euclidean(candidate , point) for candidate in xy_candidates]
    min_dist_ind = np.argmin(distances)
    offset = nearest_ind - search_radius_low 
    if return_index == 0: 
        return spline_x[min_dist_ind + offset], spline_y[min_dist_ind + offset]
    else:
        return min_dist_ind + offset



def x_on_spline(ordered, point_index):
    ref_point = ordered[0]
    length = 0
    for k in range(1, point_index + 1):
        point_0 = ordered[k-1]
        point_1 = ordered[k]
        delta = np.linalg.norm(point_1 -point_0)
        length = length + delta
    return length
    

def worm_pole(outline, end_point, other_point):
    """  """
    
    outline_pts = np.argwhere(outline == True)
    outline_pts[:,[0, 1]] = outline_pts[:,[1,0]] # Need to swap columns
    
    if len(outline_pts) == 0:
        print('Canny filter failed to find outline')
        return None
    
    vec_p = end_point - other_point 
    
    vecs_q = outline_pts - other_point
    dots_p = np.dot(vecs_q, vec_p).reshape((len(vecs_q),1))
    
    norm_p = np.linalg.norm(vec_p)
    
    norms_q = np.array([np.linalg.norm(item_q) for item_q in vecs_q]).reshape((len(vecs_q),1))
    
    cos_theta = np.divide(dots_p, np.multiply(norms_q, norm_p))
    
    
    # I need theta < pi/2. So cos(theta) needs to be positive
    maximal_cos = np.argmax(cos_theta)

    if cos_theta[maximal_cos] >0 and norms_q[maximal_cos]>norm_p:
        point = outline_pts[maximal_cos]
        return point.reshape(1,2)
    else:
        print('Finding worm pole failed')
        return None

def all_input_paths(trajectory_paths):
    with open(trajectory_paths, 'r') as f:
        paths = f.readlines()
    paths = [line.rstrip() for line in paths]
    return paths
    
def out_path(input_path, output_folder, only_name = 0):
    timelapse_n = re.findall("Timelapse/s[0-9]",input_path)[0][-1:]
    filament_n = re.findall("Filament/s[0-9]",input_path)[0][-1:]
    type_oscil = re.findall("Oscillations",input_path)
    if len(type_oscil) == 1:
        traj_type = 'Oscil'
    else:
        traj_type = 'Partial'
    out_name = f'/TL_{timelapse_n}_{traj_type}_Fil_{filament_n}.txt'
    if only_name == 1:
        return f'/TL_{timelapse_n}_{traj_type}_Fil_{filament_n}'
    else:
        return output_folder + out_name
    
    

def fancy_vx_plot(time_pts, x, v, font_size = 12, tick_size = 4, line_width = 1, dpi = 200, size=(6,3), other = [[],[],[],[]]):
    fig, ax = plt.subplots(1,1, figsize=(4,2), dpi=200)

    ax.plot(time_pts,x, '-', color='#5F2167', alpha=1, linewidth=line_width)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Position [μm]', color='#5F2167')
    ax.tick_params(size=tick_size)
    ax.tick_params(colors='#5F2167')
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    
    if other[0]!=[]:
        ax.set_yticks(other[0])
    if other[1]!=[]:
        ax.set_ylim(other[1])


    ax2 = ax.twinx() 
    ax2.set_ylabel('Speed [um/min]', color='#E35205')
    ax2.plot(time_pts, v , '-',  color='#E35205', linewidth=line_width)
    ax2.tick_params(size=tick_size)
    ax2.spines['right'].set_color('#E35205')
    ax2.spines['right'].set_linewidth(line_width)
    ax2.spines['left'].set_color('#5F2167')
    ax2.spines['left'].set_linewidth(line_width)
    ax2.spines['top'].set_linewidth(line_width)
    ax2.spines['bottom'].set_linewidth(line_width)
    ax2.tick_params(colors='#E35205')
    
    if other[2] != []:
        ax2.set_yticks(other[2])
    if other[3]!= []:
        ax2.set_ylim(other[3])

    plt.show()
    
def save_len_hists(lengths, output_folder, out_names):
    for k,leng in enumerate(lengths):
        fig, ax = plt.subplots(1,1, figsize = (5,5))
        n, bins, patches = plt.hist(leng)
        plt.ylabel('Number')
        plt.xlabel('Length (px)')
        if os.path.exists(output_folder):
            pass
        else:
            os.mkdir(output_folder)
        plt.savefig(output_folder +'//len_hists' + out_names[k] + '.png')