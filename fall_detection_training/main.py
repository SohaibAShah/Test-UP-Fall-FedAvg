import pickle
import torch
from dataset import MyDataset
from model import MyModel
from server import Server
from client import Client
from train import train_federated_model
from evaluate import evaluate_model

def main():
    # Configuration
    train_data_path = 'FL-FD/Train_data.pkl'
    test_data_path = 'FL-FD/Test_data.pkl'
    num_epochs = 200
    total_clients = 15
    num_clients_per_round = 12
    max_acc = 80.0
    classes = [f'A{i}' for i in range(1, 12)]  # Activity labels A1 to A11
    
    # Load data
    print("Loading data...")
    train_data = pickle.load(open(train_data_path, 'rb'))
    test_data = pickle.load(open(test_data_path, 'rb'))
    
    # Prepare datasets
    train_inputs = [data[0] for data in train_data.values()]  # Assuming data is [inputs, label]
    train_labels = [data[-1] for data in train_data.values()]
    test_inputs = [data[0] for data in test_data.values()]
    test_labels = [data[-1] for data in test_data.values()]
    
    train_dataset = MyDataset(train_inputs, train_labels)
    test_dataset = MyDataset(test_inputs, test_labels)
    
    # Initialize model
    print("Initializing model...")
    model = MyModel()
    model = model.double()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Initialize server and clients
    print("Initializing server and clients...")
    server = Server(model, test_dataset, num_clients_per_round)
    clients = []
    for c in range(total_clients):
        client_dataset = MyDataset([train_data[list(train_data.keys())[c]][0]], 
                                 [train_data[list(train_data.keys())[c]][-1]])
        clients.append(Client(model, {c: client_dataset}, id=c))
    
    # Train
    print("Starting training...")
    best_acc = train_federated_model(server, clients, num_epochs, num_clients_per_round, max_acc)
    
    # Evaluate
    print("Evaluating model...")
    model.load_state_dict(torch.load('model.pth'))
    evaluate_model(model, test_dataset, classes)

if __name__ == "__main__":
    main()