wget https://huggingface.co/datasets/zh-plus/tiny-imagenet/resolve/main/data/train-00000-of-00001-1359597a978bc4fa.parquet
wget https://huggingface.co/datasets/zh-plus/tiny-imagenet/resolve/main/data/valid-00000-of-00001-70d52db3c749a935.parquet

mkdir dall_e_tokenizer_weight
wget -P dall_e_tokenizer_weight https://cdn.openai.com/dall-e/encoder.pkl
wget -P dall_e_tokenizer_weight https://cdn.openai.com/dall-e/decoder.pkl

python_code='
import pandas as pd
from PIL import Image
import io
from tqdm import tqdm
import os

os.mkdir("tiny_imagenet")
train_df = pd.read_parquet("train-00000-of-00001-1359597a978bc4fa.parquet")
val_df = pd.read_parquet("valid-00000-of-00001-70d52db3c749a935.parquet")

os.mkdir("tiny_imagenet/train/")
os.mkdir("tiny_imagenet/val/")

for cl in range(200):
    os.mkdir(f"tiny_imagenet/train/{cl}")
    os.mkdir(f"tiny_imagenet/val/{cl}")

for idx in tqdm(train_df.index):
    row = train_df.loc[0]
    image = Image.open(io.BytesIO(row["image"]["bytes"]))
    image.save("tiny_imagenet/train/{}/{:d}.jpg".format(row["label"], idx))
    
for idx in tqdm(val_df.index):
    row = val_df.loc[0]
    image = Image.open(io.BytesIO(row["image"]["bytes"]))
    image.save("tiny_imagenet/val/{}/{:d}.jpg".format(row["label"], idx))
'
python -c "$python_code"
rm -r *.parquet
