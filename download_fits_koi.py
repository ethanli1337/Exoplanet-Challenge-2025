import pandas as pd
from lightkurve import search_lightcurvefile
import argparse

def download_fits_from_koi_csv(csv_path, output_dir):
    """
    Downloads FITS files for KOIs listed in a CSV using the lightkurve package.

    Args:
        csv_path (str): Path to the KOI CSV file. Must contain a 'kepid' column.
        output_dir (str): Directory to save downloaded FITS files.
    """
    df = pd.read_csv(csv_path)
    if 'kepid' not in df.columns:
        raise ValueError("CSV must contain a 'kepid' column with Kepler IDs.")

    for kepid in df['kepid']:
        try:
            search_result = search_lightcurvefile(f'Kepler-{int(kepid)}')
            if search_result:
                lcfs = search_result.download_all(download_dir=output_dir)
                print(f"Downloaded FITS for Kepler ID {kepid}")
            else:
                print(f"No FITS found for Kepler ID {kepid}")
        except Exception as e:
            print(f"Error downloading for Kepler ID {kepid}: {e}")

            if __name__ == "__main__":
                parser = argparse.ArgumentParser(description="Download FITS files for KOIs from a CSV.")
                parser.add_argument("csv_path", help="Path to the KOI CSV file.")
                parser.add_argument("output_dir", help="Directory to save downloaded FITS files.")
                args = parser.parse_args()
                download_fits_from_koi_csv(args.csv_path, args.output_dir)