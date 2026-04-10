from huggingface_hub import snapshot_download

print("Downloading TinyLlama model...")
snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_dir="models/tinyllama",
    local_dir_use_symlinks=False
)
print("Download complete. Model saved in models/tinyllama")
