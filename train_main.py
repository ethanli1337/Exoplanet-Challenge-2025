import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import BoxLeastSquares
import lightkurve as lk
from astropy.io import fits
import os
from PIL import Image
import io
from lightcurvedataset import LightCurveDataset
from exoplanet import ExoplanetCNN
from exoplanetresnet import ExoplanetResNet
from utilities import load_json_to_dict
from torch.utils.tensorboard import SummaryWriter
import datetime
from torch.utils.data import random_split
import glob
import re
from sklearn.metrics import roc_auc_score
from load_config import load_config
from sklearn.metrics import confusion_matrix
import random
from torch.utils.data import Subset

# 4. Example usage with dummy data
def create_dataset(data_folder):
    """Create a small dummy dataset for demonstration"""
    # In practice, you would have real FITS file paths and labels
    # dummy_files = ['dummy_file_1.fits', 'dummy_file_2.fits'] * 5
    # dummy_labels = [1, 0] * 5  # Alternating labels
    labels_dict = load_json_to_dict('fits_labels_dict.json')
    fits_files = []
    labels = []

    # 3. Data Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    """
    Counts ID folders under data_folder, and only plots .fits files from the first ID folder found.
    """
    id_folders = [item for item in os.listdir(data_folder)
                  if os.path.isdir(os.path.join(data_folder, item))]
    print(f"Found {len(id_folders)} ID folders in {data_folder}.")
    if not id_folders:
        print("No ID folders found.")
        return
    for id_folder in id_folders:
        id_path = os.path.join(data_folder, id_folder)
        for fname in os.listdir(id_path):
            if fname.lower().endswith('.fits'):
                fits_path = os.path.join(id_path, fname)
                fits_files.append(fits_path)
                labels.append(labels_dict.get(id_folder, 0))  # Get label for the ID folder
    return LightCurveDataset(fits_files, labels, transform=transform)

# 5. Training Function
def train_model(model, device, dataloader, criterion, optimizer, modelname, model_dir="models", num_epochs=10, log_dir="exoplanet", stop_loss=None):
    model.train()
    best_loss = float('inf')
    # Inside your training loop, before saving:
    models_dir = model_dir
    os.makedirs(models_dir, exist_ok=True)
    # Add date and time to log folder name
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is None:
        log_dir = f"exoplanet_{now}"
    else:
        log_dir = f"{log_dir}_{now}"
    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard writer

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_outputs = []

        for batch_idx, (filename, data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            # After: data, target = data.to(device), target.to(device)
            # img = data[0].cpu().numpy()  # Take the first image in the batch and move to CPU
            # # If channels are first (e.g., [3, 224, 224]), transpose to [224, 224, 3]
            # if img.shape[0] == 3:
            #     img = img.transpose(1, 2, 0)
            # # Undo normalization if needed
            # img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            # img = img.clip(0, 1)
            # plt.imshow(img)
            # plt.title("Sample input image")
            # plt.axis('off')
            # plt.tight_layout()
            # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            # plt.show()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            # print("File Name : ", filename)
            # print("Output:", output.detach().cpu().numpy())
            # print("Target:", target.detach().cpu().numpy())
            # print("Loss:", loss.item())
            running_loss += loss.item()
            global_step = epoch * len(dataloader) + batch_idx
            # Log running loss to TensorBoard
            writer.add_scalar('Batch/RunningLoss', running_loss / (batch_idx + 1), global_step)
            # Calculate accuracy
            predicted = (output.squeeze() > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
            
            # Log batch loss to TensorBoard
            writer.add_scalar('Batch/Loss', loss.item(), global_step)
            # Collect outputs and targets for AUC calculation
            # all_targets.extend(target.cpu().numpy().ravel())
            # all_outputs.extend(output.squeeze().detach().cpu().numpy().ravel())
            # try:
            #     batch_auc = roc_auc_score(target.cpu().numpy(), output.squeeze().detach().cpu().numpy())
            # except ValueError:
            #     batch_auc = float('nan')  # Not enough classes to calculate AUC
            # writer.add_scalar('Batch/AUC', batch_auc, global_step)
            # if batch_idx % 4 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        final_epoch_acc = int(epoch_acc * 10000)  # Remove decimal points
        print(f'Epoch {epoch+1} completed. Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        # # Calculate and log AUC
        # try:
        #     auc = roc_auc_score(all_targets, all_outputs)
        # except ValueError:
        #     auc = float('nan')  # Not enough classes to calculate AUC
        # print(f'Epoch {epoch+1} AUC: {auc:.4f}')
        # Log epoch loss and accuracy to TensorBoard
        writer.add_scalar('Epoch/Loss', epoch_loss, epoch)
        writer.add_scalar('Epoch/Accuracy', epoch_acc, epoch)
        # writer.add_scalar('Epoch/AUC', auc, epoch)  # <-- Log AUC here
        # Save model with loss in filename if loss improves
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join(models_dir, f"{modelname}{final_epoch_acc}_{now}.pth")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved as '{model_filename}'")
        # Stop training if loss is below threshold
        if stop_loss is not None and epoch_loss <= stop_loss:
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved as '{model_filename}'")
            print(f"Stopping early: loss {epoch_loss:.4f} reached threshold {stop_loss}")
            break
    writer.close()
    return model
def load_best_model(model_class, device, model_name, model_dir='.'):
    # Find all model files matching the pattern
    model_files = glob.glob(f"{model_dir}/{model_name}*.pth")
    best_acc = -1
    best_file = None
    pattern = re.compile(f"{model_name}(\\d+).*\\.pth")
    for file in model_files:
        filename = os.path.basename(file)
        match = pattern.search(filename)
        if match:
            acc = int(match.group(1))
            if acc > best_acc:
                best_acc = acc
                best_file = file
    if best_file is not None:
        print(f"Loading best model: {best_file} (acc={best_acc})")
        model = model_class().to(device)
        model.load_state_dict(torch.load(best_file, map_location=device))
        # model.eval()
        return model
    else:
        raise FileNotFoundError("No model files found matching pattern.")

def load_model_from_folder(model_class, device, model_name, model_dir='./models'):
    """
    Loads the latest saved model with the given model_name prefix from model_dir.
    Returns the loaded model instance.
    """
    # print("Searching in:", os.path.join(model_dir, f"{model_name}*.pth"))
    # print("Files in directory:", os.listdir(model_dir))
    model_files = glob.glob(os.path.join(model_dir, f"{model_name}.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir} with prefix '{model_name}*'")
    # Sort by modification time, newest first
    model_files.sort(key=os.path.getmtime, reverse=True)
    latest_model_file = model_files[0]
    print(f"Loading model from: {latest_model_file}")
    model = model_class().to(device)
    model.load_state_dict(torch.load(latest_model_file, map_location=device))
    return model

def prepare_data(data_path, batch_size, validation_split):
    # Create dataset (replace with your actual data)
    dataset = create_dataset(data_path)
    # Split dataset into train and test sets (e.g., 80% train, 20% test)
    train_size = int((1 - validation_split) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for train and test sets
    # Suppose labels is a list/array of all your labels (0 or 1)
    # Get the indices for the train split
    train_indices = train_dataset.indices
    labels = np.array([dataset.labels[i] for i in train_indices])  # 0 for non-exoplanet, 1 for exoplanet

    # Compute class weights (inverse frequency)
    class_sample_count = np.array([np.sum(labels == 0), np.sum(labels == 1)])
    weight = 1. / (class_sample_count + 1e-6)  # Prevent division by zero
    samples_weight = weight[labels]
    samples_weight = np.nan_to_num(samples_weight, nan=0.0, posinf=0.0, neginf=0.0)

    # Create sampler
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader
 # Example prediction function
def predict_exoplanet(test_loader, model, device):
    """
    Predict exoplanet probabilities for all samples in the test_loader.
    Prints prediction and probability for each sample.
    """
    model.eval()
    results = []
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch_idx, (filename, data, target) in enumerate(test_loader):
            data = data.to(device)
            outputs = model(data)
            probabilities = outputs.squeeze().cpu().numpy()
            for i, prob in enumerate(np.atleast_1d(probabilities)):
                prediction = "Exoplanet" if prob > 0.5 else "No exoplanet"
                print(f"Sample {batch_idx * test_loader.batch_size + i}: Prediction: {prediction} (Probability: {prob:.4f})")
                results.append(prob)
            all_preds.extend(np.atleast_1d(probabilities))
            all_true.extend(np.atleast_1d(target.cpu().numpy()))
    # After collecting all_preds (probabilities) and all_true (labels)
    binary_preds = [1 if p > 0.8 else 0 for p in all_preds]
    # cm = confusion_matrix(all_true, binary_preds)
    # Save as .npy file
    # np.save('confusion_matrix.npy', cm)
    # Optionally, save as .csv file
    # np.savetxt('confusion_matrix.csv', cm, delimiter=',', fmt='%d')
    # # Load from .npy
    # cm_loaded = np.load('confusion_matrix.npy')

    # # Or, load from .csv
    # cm_loaded_csv = np.loadtxt('confusion_matrix.csv', delimiter=',', dtype=int)
    # print("Confusion Matrix:")
    # print(cm)
    return binary_preds
    # return results
    
def main(config):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if config is not None:
        # Use config from input parameter directly
        batch_size = config.get('batch_size', 32)
        learning_rate = config.get('learning_rate', 0.001)
        num_epochs = config.get('epochs', 10)
        model_save_path = config.get('model_save_path', './models')
        data_path = config.get('data_path', './koi_data')
        validation_split = config.get('validation_split', 0.2)
        stop_loss = float(config.get('stop_loss', 5))
        weight_decay = float(config.get('weight_decay', 1e-5))
        log_dir = config.get('log_dir', 'exoplanet')
        modelname = config.get('model_name', 'exoplanet_cnn_acc_')
        train = config.get('train', False)
        fresh_start = config.get('fresh_start', False)
        print("Configuration parameters:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        exit(0)
    else :
        #load config
        config = load_config('config.yaml')
        batch_size = config.get('batch_size', 32)
        learning_rate = config.get('learning_rate', 0.001)  
        num_epochs = config.get('epochs', 10)
        model_save_path = config.get('model_save_path', './models')    
        data_path = config.get('data_path', './koi_data')
        validation_split = config.get('validation_split', 0.2)  
        stop_loss = float(config.get('stop_loss', 5))  # Optional early stopping loss threshold
        weight_decay = float(config.get('weight_decay', 1e-5))
        log_dir = config.get('log_dir', 'exoplanet')
        modelname = config.get('model_name', 'exoplanet_cnn_acc_')
        train = config.get('train', False)
        fresh_start = config.get('fresh_start', False)

    if fresh_start:
        train_loader, test_loader = prepare_data(data_path, batch_size, validation_split)
        # Get the total number of samples in the data set
        num_samples_train = len(train_loader.dataset)
        num_samples_test = len(test_loader.dataset)
        # Randomly select unique indices
        random_indices_train = random.sample(range(num_samples_train), 1000)
        random_indices_test = random.sample(range(num_samples_test), 1000)
        my_train_loader = torch.utils.data.DataLoader(
            Subset(train_loader.dataset, random_indices_train),
            batch_size=train_loader.batch_size,
            shuffle=False,
            num_workers=getattr(train_loader, 'num_workers', 0)
        )
        my_test_loader = torch.utils.data.DataLoader(
            Subset(test_loader.dataset, random_indices_test),
            batch_size=test_loader.batch_size,
            shuffle=False,
            num_workers=getattr(test_loader, 'num_workers', 0)
        )
        # Train the model
        print("Starting training from scratch...")
        model = ExoplanetResNet().to(device)
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        trained_model = train_model(model, device, my_train_loader, criterion, optimizer, modelname, model_dir=model_save_path, num_epochs=num_epochs, stop_loss=stop_loss, log_dir=log_dir)
        predict_exoplanet(my_test_loader, trained_model, device)
    else:
        try:
            model = load_best_model(ExoplanetCNN, device, modelname, model_dir=model_save_path)
            # model = load_model_from_folder(ExoplanetResNet, device, 'exoplanet_cnn_acc_9985_20250924_073114', model_dir='./models')
        except Exception as e:
            print(f"Error loading model, start from scratch: {e}")
            model = ExoplanetCNN().to(device)
            # model = load_model_from_folder(ExoplanetResNet, device, 'exoplanet_cnn_acc_9985_20250924_073114', model_dir='./models')

        if train:
            train_loader, test_loader = prepare_data(data_path, batch_size, validation_split)
            # Train the model
            print("Starting training...")
            criterion = nn.BCELoss()  # Binary Cross Entropy Loss
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            trained_model = train_model(model, device, train_loader, criterion, optimizer, modelname, num_epochs=num_epochs, model_dir=model_save_path, stop_loss=stop_loss, log_dir=log_dir)
            predict_exoplanet(test_loader, trained_model, device)
        else:
            _, test_loader = prepare_data(data_path, batch_size, 0)
            predict_exoplanet(test_loader, model, device)

    
    
    
    # Initialize model, loss, and optimizer
    # model = ExoplanetCNN().to(device)
    # using ResNet18
    # model = ExoplanetResNet().to(device)

# 6. Main execution
if __name__ == "__main__":
    main(config=None)
    
 