import torch
from train_main import predict_exoplanet
from train_main import load_model_from_folder
from train_main import prepare_data
from exoplanetresnet import ExoplanetResNet
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from lightcurvedataset import LightCurveDataset
import random
import tempfile

def predict(fits):
    # Placeholder imports for your dataset and model
    # from your_dataset_module import YourDataset
    # from your_model_module import YourModel

    # parser = argparse.ArgumentParser(description="Predict exoplanets using a trained model.")
    # parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model file')
    # parser.add_argument('--test-data-path', type=str, required=True, help='Path to the test data file')
    # parser.add_argument('--batch-size', type=int, default=32, help='Batch size for DataLoader')
    # args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_folder(ExoplanetResNet, device, 'exoplanet_cnn_acc_9985', model_dir='./models')
    if fits is None:
        # Load your test dataset
        train_loader, test_loader = prepare_data('koi_data', 32, 0.2)
        # Get the total number of samples in the test set
        num_samples = len(test_loader.dataset)
        # Randomly select 1000 unique indices
        random_indices = random.sample(range(num_samples), 100)
        my_test_loader = torch.utils.data.DataLoader(
            Subset(test_loader.dataset, random_indices),
            batch_size=test_loader.batch_size,
            shuffle=False,
            num_workers=getattr(test_loader, 'num_workers', 0)
        )
        predict_exoplanet(my_test_loader, model, device)
    else:
         # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp_file:
            tmp_file.write(fits.read())
            tmp_fits_path = tmp_file.name
        # Example: create a dataset with a single FITS file
        fits_file = tmp_fits_path  # Replace with your FITS file path
        label = 1  # or 0, depending on your use case

        fits_files = [fits_file]
        labels = [label]
        transform = transforms.ToTensor()  # Add this line
        dataset = LightCurveDataset(fits_files, labels, transform=transform)
        p_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        # To get the processed record (image and label) from the dataset:
        # fits_path, image, label_tensor = dataset[0]
        # print(f"FITS path: {fits_path}")
        # print(f"Image shape: {image.size if hasattr(image, 'size') else image.shape}")
        # print(f"Label: {label_tensor}")
        predicts = predict_exoplanet(p_loader, model, device)

if __name__ == "__main__":
    predict(fits=None)