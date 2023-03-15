import os
import chess.pgn
import re
import requests
import zstandard as zstd
from transformers import (
    TFGPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    TFTrainer,
    TFTrainingArguments,
)

def preprocess_pgn_file(pgn_file_path, output_file_path):
    with open(pgn_file_path, 'r') as pgn_file, open(output_file_path, 'w') as output_file:
        game_count = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            main_line = game.mainline_moves()
            moves_san = " ".join([move.uci() for move in main_line])
            output_file.write(moves_san + '\n')
            game_count += 1
            if game_count % 1000 == 0:
                print(f"Processed {game_count} games")
    print(f"Finished processing {game_count} games")

# Download and decompress the dataset
url = "https://database.lichess.org/standard/lichess_db_standard_rated_2014-07.pgn.zst"
compressed_file = "lichess_db_standard_rated_2014-07.pgn.zst"
decompressed_file = "lichess_db_standard_rated_2014-07.pgn"

print("Downloading the dataset...")
response = requests.get(url, stream=True)
with open(compressed_file, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
print("Download complete")

print("Decompressing the dataset...")
dctx = zstd.ZstdDecompressor()
with open(compressed_file, 'rb') as ifh, open(decompressed_file, 'wb') as ofh:
    dctx.copy_stream(ifh, ofh)
print("Decompression complete")

# Preprocess the dataset
preprocessed_file = "preprocessed_games.txt"
print("Preprocessing the dataset...")
preprocess_pgn_file(decompressed_file, preprocessed_file)
print("Preprocessing complete")

# Prepare the dataset for training
print("Preparing the dataset for training...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=preprocessed_file,
    block_size=128,
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
print("Dataset preparation complete")

# Fine-tune a pre-trained transformer model
print("Fine-tuning the model...")
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

training_args = TFTrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = TFTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
print("Model training complete")

# Save the trained model
trainer.save_model('./trained_model')

# Generate new chess games
# model = TFGPT2LMHeadModel.from_pretrained('./trained_model')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
