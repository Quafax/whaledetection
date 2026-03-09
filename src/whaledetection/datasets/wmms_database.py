from datasets import load_dataset
import os
#watkins marine mamal database is at the moment not reachable. So we download a part of it from huggingface
#transform the bytes from wmms huggingface archive to wav files, because of torchcodec errors and to keep the original sampling rate
# also reachable via https://web.archive.org/web/20240725193523/https://cis.whoi.edu/science/B/whalesounds/
def sanitize_folder_name(name: str) -> str:
    bad_chars = r'<>:"/\|?*'
    for ch in bad_chars:
        name = name.replace(ch, "_")
    return name


def export_split(base_out_dir: str, split: str) -> None:
    ds = load_dataset("confit/wmms-parquet", split=split)

    try:
        table = ds.data
    except AttributeError:
        table = ds._data

    audio_col = table.column("audio")
    species_col = table.column("species")

    split_out_dir = os.path.join(base_out_dir, split)
    #that does make it slow probably but is really okay for that size of database
    #if used for bigger try remembering the folder
    os.makedirs(split_out_dir, exist_ok=True)

    num_rows = table.num_rows

    for i in range(num_rows):
        audio_info = audio_col[i].as_py()
        audio_bytes = audio_info["bytes"]
        orig_name = audio_info.get("path", f"sample_{i}.wav")

        species_name = species_col[i].as_py()
        species_folder = sanitize_folder_name(species_name)

        out_dir = os.path.join(split_out_dir, species_folder)
        os.makedirs(out_dir, exist_ok=True)

        orig_basename = os.path.basename(orig_name)
        out_filename = f"{i:06d}_{orig_basename}"
        out_path = os.path.join(out_dir, out_filename)

        with open(out_path, "wb") as f:
            f.write(audio_bytes)

        if i % 100 == 0:
            print(f"[{split}] [{i}/{num_rows}] saved: {out_path}")


def export_wmms_database(base_out_dir: str) -> None:
    #export both splits
    export_split(base_out_dir, "train")
    export_split(base_out_dir, "test")

