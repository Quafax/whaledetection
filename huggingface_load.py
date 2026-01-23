
from huggingface_hub import login
login()
#dataset = load_dataset("confit/wmms-parquet")
from datasets import load_dataset
train_dataset = load_dataset("confit/wmms-parquet", split="train")
print(train_dataset.shape)
example = train_dataset[0]              # erstes Sample
audio = example["audio"]     # Audio-Feature