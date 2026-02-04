from huggingface_hub import login
login()
from datasets import load_dataset
import os

#####the bytes are directly transformed to wav files, so we can evade torchcodec issues 
# and the torchcodec sr of 16000 for this dataset


# set here target output directory for all the wav files
base_out_dir = r"C:\Users\Admin\WhaleDetection\export_whales"

####### can do this with test split and train split
ds = load_dataset("confit/wmms-parquet", split="train")

try:
    table = ds.data
except AttributeError:
    table = ds._data

audio_col   = table.column("audio")
species_col = table.column("species")

def sanitize_folder_name(name: str) -> str:
    bad_chars = r'<>:"/\|?*'
    for ch in bad_chars:
        name = name.replace(ch, "_")
    return name

os.makedirs(base_out_dir, exist_ok=True)

num_rows = table.num_rows

for i in range(num_rows):
    audio_info = audio_col[i].as_py()
    audio_bytes = audio_info["bytes"]
    orig_name   = audio_info.get("path", f"sample_{i}.wav")

    species_name   = species_col[i].as_py()
    species_folder = sanitize_folder_name(species_name)

    out_dir = os.path.join(base_out_dir, species_folder)
    os.makedirs(out_dir, exist_ok=True)

    orig_basename = os.path.basename(orig_name)
    out_filename  = f"{i:04d}_{orig_basename}"
    out_path      = os.path.join(out_dir, out_filename)

    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    if i % 100 == 0:
        print(f"[{i}/{num_rows}] gespeichert: {out_path}")
