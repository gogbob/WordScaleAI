import kagglehub

# Download latest version
path = kagglehub.dataset_download("shubchat/1002-short-stories-from-project-guttenberg")

print("Path to dataset files:", path)