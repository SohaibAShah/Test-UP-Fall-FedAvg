import numpy as np
def fuse_data(GAF_data, Camera_data):
    """Combine GAF-transformed sensor data and camera data.
    
    Args:
        GAF_data (dict): Sensor data with GAF transformations.
        Camera_data (dict): Processed camera data.
    
    Returns:
        dict: Fused data with keys (Subject, Activity, Trial) and values as lists of fused arrays.
    """
    combined_keys = list(set(GAF_data.keys()) & set(Camera_data.keys()))  # Ensure matching keys
    combined_keys.sort()
    GAF_Camera_data = {}
    
    for key in combined_keys:
        Ankle = np.concatenate((GAF_data[key]['GAF_Ankle'].transpose(2, 0, 1), Camera_data[key]), axis=0)
        RightPocket = np.concatenate((GAF_data[key]['GAF_RightPocket'].transpose(2, 0, 1), Camera_data[key]), axis=0)
        Belt = np.concatenate((GAF_data[key]['GAF_Belt'].transpose(2, 0, 1), Camera_data[key]), axis=0)
        Neck = np.concatenate((GAF_data[key]['GAF_Neck'].transpose(2, 0, 1), Camera_data[key]), axis=0)
        Wrist = np.concatenate((GAF_data[key]['GAF_Wrist'].transpose(2, 0, 1), Camera_data[key]), axis=0)
        l = [Ankle, RightPocket, Belt, Neck, Wrist]
        GAF_Camera_data[key] = l
    
    return GAF_Camera_data