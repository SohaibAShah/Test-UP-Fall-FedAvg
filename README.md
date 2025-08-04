# Test-UP-Fall-FedAvg
Implemenattion of the Federated Learning using UP-Fall Dataset

dataset/
├── CompleteDataSet.csv
├── downloaded_camera_files/
│   ├── Subject1Activity1Trial1Camera1.zip
│   ├── Subject1Activity1Trial1Camera2.zip
│   ├── ...
├── camera/  # Temporary directory for extracted images (created/deleted during processing)
├── Train_data.pkl
├── Test_data.pkl
├
fall_detection_preprocessing/
├── utils.py
├── sensor_processing.py
├── camera_processing.py
├── data_fusion.py
├── dataset_splitting.py
├── main.py
├
fall_detection_training/
├── dataset.py
├── model.py
├── server.py
├── client.py
├── train.py
├── evaluate.py
├── main.py