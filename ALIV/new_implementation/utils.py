import numpy as np
import os
import sys
from tqdm import tqdm
import pickle
#from scipy import ndimage as snp
#from statsmodels.tsa.stattools import acf
from natsort import natsorted
import cv2
from numpy.fft import fft2,fft,ifft
from skimage.transform import warp, AffineTransform
from scipy.optimize import curve_fit
# import cupy as cp

def min_max(data1):
    maxx = np.max(data1)
    if maxx==0:
        return data1
    else:
        data1 = (data1-np.min(data1))/(maxx-np.min(data1))
        return data1

def load_nested_data_pickle(path, num):
    pic_paths = []
    for scan_num in natsorted(os.listdir(path)):
        if scan_num.startswith('scan'):
            pic_paths.append(os.path.join(path,scan_num,f'{scan_num}.pickle'))
    pic_paths = pic_paths[0:num]
    with open(f'{pic_paths[0]}', 'rb') as handle:
        b = pickle.load(handle)
    data = np.zeros((len(pic_paths),b.shape[0],b.shape[1],b.shape[2]))

    for idx,img_path in enumerate(pic_paths):
        with open(img_path, 'rb') as handle:
            temp = pickle.load(handle)
        data[idx]=(temp.copy())
    data = data.astype(np.float64)
    return data

def concatenate_scan_set(nest_set):
    #transform nest data into 3D concate data
    data = np.zeros((nest_set.shape[0]*nest_set.shape[1],nest_set.shape[2],nest_set.shape[3]))
    for scan in range(nest_set.shape[0]):
        data[scan*nest_set.shape[1]:(scan+1)*nest_set.shape[1], :, :]=nest_set[scan, :, :, :].copy()
    data = data.astype(np.float64)
    return data



def load_nested_data_png(path):
    pic_paths = []
    for scan_num in os.listdir(path):
        if scan_num.startswith('scan'):
            pic_paths.append(os.path.join(path,scan_num))
    pic_paths = natsorted(pic_paths)
    print(os.path.join(pic_paths[0],os.listdir(pic_paths[0])[0]))
    # print(pic_paths[0]+os.listdir(pic_paths[0])[0])
    temp_img = cv2.imread(os.path.join(pic_paths[0],os.listdir(pic_paths[0])[0]),cv2.IMREAD_UNCHANGED) 
    data = np.zeros((len(pic_paths),len(os.listdir(pic_paths[0])),temp_img.shape[0],temp_img.shape[1]))

    for main_idx,img_paths in enumerate(pic_paths):
        all_img_paths = natsorted(os.listdir(img_paths))
        for idx,img_path in enumerate(all_img_paths):
            temp = cv2.imread(os.path.join(img_paths,img_path),cv2.IMREAD_UNCHANGED)
            data[main_idx,idx]=(temp.copy())
    data = data.astype(np.float64)
    return data

def load_data_png(path):
    pic_paths = []
    for i in os.listdir(path):
        if i.endswith('.PNG') or i.endswith('.png'):
            pic_paths.append(i)
    pic_paths = natsorted(pic_paths)

    temp_img = cv2.imread(path+pic_paths[0],cv2.IMREAD_UNCHANGED) 
    imgs_from_folder = np.zeros((len(pic_paths),temp_img.shape[0],temp_img.shape[1]))
    # imgs_from_folder = []
    for i,j in enumerate(pic_paths):
        aa = cv2.imread(path+j,cv2.IMREAD_UNCHANGED)
        imgs_from_folder[i] = aa.copy()
    imgs_from_folder = imgs_from_folder.astype(np.float64)
    return imgs_from_folder

def slope_mask(slope_arr):
    mask1 = np.zeros_like(slope_arr[0],dtype=np.float64)
    std_mask = np.apply_along_axis(func1d=np.std,arr=slope_arr,axis=0)
    slope_arr = np.apply_along_axis(func1d=min_max,arr=slope_arr,axis=0)
    for x in range(slope_arr.shape[1]):
        for y in range(slope_arr.shape[2]):
            data1 = slope_arr[:,x,y]
            slope1 = np.polyfit(range(len(data1)), data1, 1)[0]
            mask1[x, y] = -np.abs(slope1)
    return mask1*(5*std_mask)

def ymotion(data):
    # n = data.shape[0]
    nn = [np.argmax(np.sum(data[i][0,:,:],axis=1)) for i in range(data.shape[0])]
    tf_all_nn = np.tile(np.eye(3),(data.shape[0],1,1))
    for i in range(tf_all_nn.shape[0]):
        tf_all_nn[i] = np.dot(tf_all_nn[i],AffineTransform(translation=(0,-(nn[0]-nn[i]))))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j]  = warp(data[i][j],AffineTransform(matrix=tf_all_nn[i]),order=3)
    return data



## aLIV related function
def AVG_LIV(oneD_data, time_index_list):
    log_data = 10 * np.log10(oneD_data + 1e-8)#convert the intensity to log data, add small number to prevent 0 from occuring  
    time_avgLIV = np.zeros((len(time_index_list), 2))#data structure to store time interval and its avgLIV
    for i in range(len(time_index_list)):
        index_data_group = time_index_list[i][1]
        num_data_group = len(index_data_group)#number of data groups that belong to this specific time interval
        LIV_group = np.zeros(num_data_group)
        for j in range(num_data_group):
            sub_log_data = log_data[index_data_group[j]]
            avglog = np.sum(sub_log_data) / (np.max(index_data_group[j])-np.min(index_data_group[j]))#time average of LIV
            sub_log_substraction = sub_log_data - avglog
            sub_LIV = np.mean(np.square(sub_log_substraction))#LIV of this particular sub dataset
            LIV_group[j] = sub_LIV
        time_avgLIV[i, :] = [time_index_list[i][0], np.mean(LIV_group)]#the average LIV of this particular time interval
    return time_avgLIV


def LIV_fun(Tw, a, tau):
    return a * (1 - np.exp(-Tw / tau))

#function in multiprocessing


def makeSparseDataFromRasterRepeat(octFrames, lpg, fpl, floc):
    """
    This function extracts time-sequence at a particular location from
    a raster-repeating data volume (Ibrahim2021BOE's volume).
    Parameters
    ----------
    octFrames : 3D double [timePoints, z, x] or [timePoints, x, y]
        Time sequnce OCT intensity taken by repeating raster scan protocol (Ibrahim2021BOE).
    lpg : int
        Locations per group 
    fpl : int
        frames per location
    floc : int
        Frame Location of Interest.
        The data sequence at this location is returned as 'sparseSequence'.
        
    Returns
    -------
    sparseSequence : 3D double [timePoints, x, z]
        Time sequence OCT intensity under speudo-sparse acquisition.
    timePoints : 1D int array in original frame index
        The sequence of time poins at which the sparseSequence was acquired.
    """

    fpg = lpg*fpl # frames/group #25000 (lpg 500 * fpl 50)
    # floc 0,1,2,3,....
    theGrp = int(floc/lpg) # The group containing the location (loc) # ~0
    fIdxInG = floc - theGrp*lpg # The frame index in the group # 0,1,2,3,.....

    fStart = fpg * theGrp + fIdxInG # The start frame index in the volume of the location. # 0,1,2,3....
    fStop = fStart + (fpg - fIdxInG) - 1  # 24999
    frameIndexes = range(int(fStart), int(fStop) + 1, int(lpg)) # 0,500,1000.....24500
    timePoints = np.linspace(0,fpl-1, fpl, dtype = int)*lpg # 0,500,1000.....24500

    sparseSequence = np.array(octFrames[np.array(frameIndexes).astype(int) , :, :])
    return(sparseSequence, timePoints)

def seekDataForMaxTimeWindow(timePoints, mtw):
    """
    Compute valid combination of timePoints for which the maximum timepoint is smaller than mtw. 
    The valid time sequence will be used to extract
        the OCT frames within the particular time window from whole repeating frames.
        
    Parameters
    ----------
    timePoints : 1D numpy array
        Sequence of time points at which OCT data was taken [in frame time unit]
    mtw : 1D array, int
        The set maximum time window [in frame time unit]

    Returns
    -------
    validTimePointMatrix: 2D numpy array, bool.
        Valid time sequence. VTS[i,:] is the i-th valid time sequence.
    """
    A = np.ones((1,timePoints.shape[0]))
    B = timePoints.reshape(timePoints.shape[0],1)
    timePointMatrix = np.transpose(A*B) - A*B
    validTimePointMatrix = (timePointMatrix <= mtw)*(timePointMatrix >= 0)

    ##---------------------
    ## Let's rewirte later as not to use for-loop
    ##---------------------    
    trueMtw = np.zeros(validTimePointMatrix.shape[0])
    for i in range (0,validTimePointMatrix.shape[0]):
        X = validTimePointMatrix[i,:]
        X = timePoints[X]
        Y = np.max(X) - np.min(X)
        trueMtw[i] = Y
    
    validTimePointMatrix = validTimePointMatrix[(trueMtw >= mtw),:]
    # validTimePointMatrix = np.asnumpy(validTimePointMatrix)
    return(validTimePointMatrix)

def computeVLIV(OctSequence, timePoints, maxTimeWidth = np.nan, compute_VoV = False):
    """
    compute LIV curve (VLIV) from time sequential OCT linear intensity.

    Parameters
    ----------
    OctSequence : double (time, x, z) or (time, z, x)
        Time sequence of linear OCT intensity.
        It can be continuous sequence or sparse time sequence.
        
    timePoints : 1D int array (the same size with time of OctSequence)
        indicates the time point at which the frames in the OCT sequence were taken.

    maxTimeWidth : int
        The maximum time width for LIV computation.
        If the LIV curve fitting uses only a particular time-region of LIV curve, 
        it is unnnessesary to compute LIV at the time region exceeding the fitting region.
        With this option, you can restrict the LIV computation time-region and
        can reduce the computation time.
        The default is NaN. If it is default, i.e., maxTimeWidth was not set,
        the full width will be computed.
    compute_VoV: True or False
        Compute variance of all LIVs of identical time window (VoV) : True
        Don't compute VoV : False
        
    Returns
    -------
    VLIV : 3D double (Time points, x, z) or (Time points, z, x)

    possibleMtw : 1D array, int        
        Time points correnspinding to the max-time-window axis of VLIV.
        
    VoV : 3D double (max time window, x, z)
        variance of variances (LIVs) 
    """
    
    
    # Compute all possible maximum time window
    timePoints = np.asarray(timePoints)
    A = np.ones((1,timePoints.shape[0]))
    B = np.asarray(timePoints.reshape(timePoints.shape[0],1))
    timePointMatrix = np.transpose(A*B) - A*B
    timePointMatrix[timePointMatrix<0] = 0.
    possibleMtw = np.unique(timePointMatrix)
    possibleMtw = possibleMtw[possibleMtw != 0.] # to remove the elemnt of 0.0
    
    # Reduce the time-region to be computed to meet with "maxTimeWidth"
    if np.isnan(maxTimeWidth):
        pass
    else:
        maxTimeWidth = np.asarray(maxTimeWidth)
        possibleMtw = possibleMtw[possibleMtw <= maxTimeWidth] 
    
    logSparseSequence = 10*np.log10(OctSequence + 1e-8)
    logSparseSequence = np.asarray(logSparseSequence)             # for cupy computation
    VLIV = np.zeros((possibleMtw.shape[0], logSparseSequence.shape[1], logSparseSequence.shape[2]))

    # variance of variance
    VoV = np.zeros((possibleMtw.shape[0], logSparseSequence.shape[1], logSparseSequence.shape[2]))

    i = 0        
    for mtw in possibleMtw:
        
        validTimePointMatrix = seekDataForMaxTimeWindow(timePoints, mtw)
        validTimePointMatrix = np.asarray(validTimePointMatrix)    # for cupy computation 
        
        if compute_VoV == True:
            Var = np.zeros((validTimePointMatrix.shape[0], logSparseSequence.shape[1], logSparseSequence.shape[2] ))       
        # cupy compute
        for j in range(0, validTimePointMatrix.shape[0]):
            VLIV[i] = VLIV[i] + np.nanvar(logSparseSequence[validTimePointMatrix[j,:]],axis=0)
            # newly added for LIV for each subset 
            if compute_VoV == True:
                Var[j] = np.nanvar(logSparseSequence[validTimePointMatrix[j,:]],axis=0)

        VLIV[i] = VLIV[i] / (validTimePointMatrix.shape[0])
        
        # variance of variance (VoV) at a single time window (2D array)
        if compute_VoV == True:
            VoV[i] =np.var(Var, axis=0)
        
        i = i+1
    
    # VLIV = np.asnumpy(VLIV)   # for numpy computation
    # possibleMtw = np.asnumpy(possibleMtw)
    # VoV = np.asnumpy(VoV)
    return(VLIV, possibleMtw, VoV)


def vlivCPUFitExp(VLIV, possibleMtw, frameSeparationTime, alivInitial, swiftInitial,
                  bounds = ([0,0],[np.inf, np.inf]), use_constraint = False):
    """
    Provide saturation level (magnitude) and time constant (tau)
    from LIV curve (VLIV) by exponential npU-based fitting.

    Parameters
    ----------
    VLIV : 3D double array, (time window, z, x)
        LIV curve (VLIV)
    possibleMtw : 1D int array
        time window indicators for VLIV data array.
    frameSeparationTime: constant (float)
        Successive frame measurement time [s] 
    alivInitial: float
        alivInitial = Initial value of a in fitting
    swiftInitial: float
        1/swiftInitial = Initial value of b in fitting
    bounds : 2D tuple
        bounds for fitting
        ([min a, min b], [max a, max b])
    use_constraint : True or False
        Set bounds of parameters in fitting : True
        don't set bounds : False

    Returns
    -------
    mag : 2d double (z, x)
        magnitude of the 1st order saturation function.
    tau : 2d doubel (z, x)
        time constant of the 1st-order saturation function.
    
    """
    height = VLIV.shape[1]
    width = VLIV.shape[2]
    mag = np.empty((height, width))
    tau = np.empty((height, width))

    def saturationFunction(x, a, b):
        return(np.absolute(a)*(1-(np.exp(-x/b))))
    
    for depth in range(0, int(height)):
      
        for lateral in range(0, int(width)):
            LivLine = VLIV[:,depth, lateral]
            ## Remove nan from LivLine (and also from corresponding possibleMtw).
            nonNanPos = (np.isnan(LivLine) == False)
            if np.sum(nonNanPos) >= 2:
                LivLine2 = LivLine[nonNanPos]
                t = possibleMtw[nonNanPos]
                ## 1D list of time window in frame units -> 1D list of time window in second unit.
                for i in range (len(t)):
                    t[i] = t[i] * frameSeparationTime

                try:
                    if use_constraint == False:
                        popt = curve_fit(saturationFunction, 
                                           t,
                                           LivLine2,
                                            method = "lm", # when no boundary, "lm"
                                           p0 = [alivInitial, 1/swiftInitial] )[0]
                    else: 
                        popt = curve_fit(saturationFunction, 
                                           t,
                                           LivLine2,
                                            method = "dogbox",# when add boundary, "dogbox"
                                           p0 = [alivInitial, 1/swiftInitial],
                                           bounds = bounds)[0]# set boundary [min a, min b],[max a, max b]
                except RuntimeError:
                    mag[depth, lateral] = np.nan
                    tau[depth, lateral] = np.nan
                    
                mag[depth, lateral] = popt[0]
                tau[depth, lateral] = popt[1]

            else:
                mag[depth, lateral] = np.nan
                tau[depth, lateral] = np.nan
    
    return(mag, tau)