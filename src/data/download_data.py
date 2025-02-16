# src/data/download_data.py

import os
import urllib.request
import tarfile
import gzip
import shutil
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download a file from URL to the specified output path with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_wmt14_en_de(data_dir="data/raw"):
    """
    Download and extract WMT14 English-German dataset.
    The dataset includes:
    - Europarl v7
    - Common Crawl corpus
    - News Commentary
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Define dataset URLs
    urls = {
        'europarl': 'https://www.statmt.org/europarl/v7/de-en.tgz',
        'common_crawl': 'https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
        'news_commentary': 'https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz'
    }
    
    # Download and extract each dataset
    for dataset_name, url in urls.items():
        print(f"\nDownloading {dataset_name}...")
        
        # Download
        output_file = os.path.join(data_dir, f"{dataset_name}.tgz")
        if not os.path.exists(output_file):
            download_url(url, output_file)
        
        # Extract
        print(f"Extracting {dataset_name}...")
        with tarfile.open(output_file) as tar:
            tar.extractall(path=data_dir)
        
        # Clean up
        os.remove(output_file)
    
    print("\nDownload complete! Dataset is ready in", data_dir)

if __name__ == "__main__":
    download_wmt14_en_de()
