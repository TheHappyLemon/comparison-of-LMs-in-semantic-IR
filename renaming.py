import os

def rename_folders(root_directory):
    # Walk through all subdirectories starting from the root directory
    for dirpath, dirnames, filenames in os.walk(root_directory):
        if "source" in dirnames:
            old_path = os.path.join(dirpath, "source")
            new_path = os.path.join(dirpath, "full-text")
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

        if "open" in dirnames:
            old_path = os.path.join(dirpath, "open")
            new_path = os.path.join(dirpath, "introduction")
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

root_directory = r"C:\Univer\comparison-of-LMs-in-semantic-IR"
rename_folders(root_directory)
