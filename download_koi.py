import requests

def download_kepler_koi_csv():
    url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&select=kepid,kepoi_name,koi_disposition,koi_score&format=csv"
    response = requests.get(url)
    with open("kepler_koi.csv", "wb") as f:
        f.write(response.content)
    print("Download complete: kepler_koi.csv")

if __name__ == "__main__":
    download_kepler_koi_csv()