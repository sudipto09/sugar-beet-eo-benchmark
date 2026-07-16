from huggingface_hub import snapshot_download

print("Downloading Prithvi EO 2.0 300M pretrained weights...")
path = snapshot_download(
    repo_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
    local_dir="/home/sudiptochakraborty/praktikum_cv/models/prithvi_300m"
)
print(f"Done. Downloaded to: {path}")

# Verify key files exist
import os
files = os.listdir(path)
print(f"Files downloaded: {files}")
