import streamlit as st
import yaml
from train_main import main
from predict import predict

def create_raw_fft_images(fits_file):
    """
    Given a FITS file (path or file-like object), returns a matplotlib figure or PIL image
    of the raw and/or FFT light curve.
    """
    from astropy.io import fits
    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image

    # Read FITS file (handle both path and file-like object)
    if hasattr(fits_file, "read"):
        hdul = fits.open(fits_file)
    else:
        hdul = fits.open(str(fits_file))

    data = hdul[1].data
    time = data['TIME']
    flux = data['PDCSAP_FLUX']
    hdul.close()

    # Remove NaNs
    mask = ~np.isnan(time) & ~np.isnan(flux)
    time = time[mask]
    flux = flux[mask]

    fft_vals = np.fft.fft(flux)
    fft_freq = np.fft.fftfreq(len(flux), d=np.median(np.diff(time)))

    pos_mask = fft_freq > 0
    freqs = fft_freq[pos_mask]
    amplitudes = np.abs(fft_vals[pos_mask])
    # Plot raw light curve
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(time, flux, 'k.', markersize=1, alpha=0.7)
    ax[0].set_xlabel("Time (BJD)")
    ax[0].set_ylabel("Flux")
    ax[0].set_title("Raw Light Curve")

    # Plot FFT
    ax[1].plot(freqs, amplitudes)
    ax[1].set_xlabel('Frequency')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_title('FFT Spectrum')
    plt.tight_layout()

    # Convert plot to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    return image
# Helper: Load config from YAML
def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

# Helper: Save config to YAML (optional)
def save_config(config, config_file='config.yaml'):
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f)

# Streamlit UI for ML exoplanet pipeline
st.title("Exoplanet ML Training Dashboard")

config = load_config('config.yaml')

st.sidebar.header("Training Configuration")

batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=1024, value=config.get('batch_size', 32))
learning_rate = st.sidebar.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=float(config.get('learning_rate', 0.001)), format="%.6f")
num_epochs = st.sidebar.number_input("Epochs", 1, 100, value=int(config.get('epochs', 10)))
model_save_path = st.sidebar.text_input("Model save path", config.get('model_save_path', './models'))
data_path = st.sidebar.text_input("Data path", config.get('data_path', './koi_data'))
validation_split = st.sidebar.slider("Validation split", 0.01, 0.99, float(config.get('validation_split', 0.2)))
stop_loss = st.sidebar.number_input("Early stopping loss threshold", min_value=0.0, max_value=100.0, value=float(config.get('stop_loss', 5)))
weight_decay = st.sidebar.number_input("Weight decay", min_value=0.0, max_value=1.0, value=float(config.get('weight_decay', 1e-5)), format="%.6f")
log_dir = st.sidebar.text_input("Log directory", config.get('log_dir', 'exoplanet'))
modelname = st.sidebar.text_input("Model name base", config.get('model_name', 'exoplanet_cnn_acc_'))
train = st.sidebar.checkbox("Train mode", config.get('train', False))
fresh_start = st.sidebar.checkbox("Fresh start", config.get('fresh_start', False))

if st.button("Run Training/Infernence"):
    # Build config dict for main()
    curr_config = {
        'batch_size': batch_size,
        'learning_rate': float(learning_rate),
        'epochs': num_epochs,
        'model_save_path': model_save_path,
        'data_path': data_path,
        'validation_split': validation_split,
        'stop_loss': float(stop_loss),
        'weight_decay': float(weight_decay),
        'log_dir': log_dir,
        'model_name': modelname,
        'train': train,
        'fresh_start': fresh_start,
    }
    # Optionally save to YAML
    save_config(curr_config, 'config.yaml')

    # Call your main logic here, passing curr_config as needed
    main(curr_config)
    st.success("Parameters saved and model run triggered (add logic to call main() here).")

# --- Add FITS file uploader and run button below ---
st.header("Upload FITS File for Inference")
uploaded_fits = st.file_uploader("Choose a FITS file", type=["fits"])

if uploaded_fits is not None:
    st.info(f"Uploaded file: {uploaded_fits.name}")
    if st.button("Run Inference on Uploaded FITS"):
        # Add your inference logic here, e.g., call a function to process the FITS file
        predcits = predict(uploaded_fits)
         # Assume predict returns 1 if it has a curve, 0 if not
        if predcits == 1 or (isinstance(predcits, (list, tuple)) and predcits[0] == 1):
            st.success(f"Inference run on {uploaded_fits.name}: **It has a curve.**")
        else:
            st.success(f"Inference run on {uploaded_fits.name}: **It does not have a curve.**")
    if st.button("Plot Uploaded FITS"):
        image = create_raw_fft_images(uploaded_fits)
        st.image(image, caption=f"FFT/Raw Light Curve for {uploaded_fits.name}", use_container_width=True)
        # Add your inference logic here, e.g., call a function to process the FITS file
        # st.success(f"Inference run on {uploaded_fits.name} (add logic to process the file here).")        
# Optionally: File uploader to load different config.yaml files
# uploaded_file = st.sidebar.file_uploader("Upload YAML config", type=["yaml", "yml"])
# if uploaded_file is not None:
#     config = yaml.safe_load(uploaded_file)
#     st.experimental_rerun()  # Force page to refresh with new config
