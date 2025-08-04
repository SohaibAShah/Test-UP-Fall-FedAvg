from sensor_processing import process_sensor_data
from camera_processing import process_camera_data
from data_fusion import fuse_data
from dataset_splitting import split_and_save_data

def main():
    # Configuration
    sensor_path = 'dataset'
    camera_zip_path = '/home/syed/PhD/Fall-Detection-Research-1/UP-Fall Dataset/downloaded_camera_files'
    temp_camera_path = 'dataset/camera'
    output_dir = 'dataset'
    
    # Process sensor data
    print("Processing sensor data...")
    GAF_data = process_sensor_data(sensor_path)
    
    # Process camera data
    print("Processing camera data...")
    Camera_data = process_camera_data(camera_zip_path, temp_camera_path)
    
    # Fuse data
    print("Fusing sensor and camera data...")
    GAF_Camera_data = fuse_data(GAF_data, Camera_data)
    
    # Split and save data
    print("Splitting and saving data...")
    Train_data, Test_data = split_and_save_data(GAF_Camera_data, output_dir)

if __name__ == "__main__":
    main()