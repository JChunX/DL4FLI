from pathlib import Path
import os
import zipfile


def extract_link(data_dir, link):
    """
        Given a directory and zip link, 
        download and extract to directory
    """
    if not os.path.exists(data_dir):
        print('\nCreating data directory..')
        os.makedirs(data_dir)

    zip_path = link.split('/')[-1]
    zip_path = os.path.join(data_dir,zip_path)
    extracted_path = os.path.join(data_dir,zip_path.split('.')[0])

    if not Path(extracted_path).exists():
        print('\nExtracting data to: {}'.format(zip_path))
        os.system('wget -O {0} {1}'.format(zip_path, link)) 
        with zipfile.ZipFile(zip_path, 'r') as f:
            f.extractall(data_dir)
    else:
        print('\nFile already exists! Stopping..')
    return extracted_path