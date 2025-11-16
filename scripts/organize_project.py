import os
import shutil

# Define project root
project_root = r"C:\Users\anush\OneDrive\Desktop\Documents\supermart_project"

# Define folders to organize files
folders = {
    "data": ["csv"],  # dataset files
    "scripts": ["py"],  # python scripts
    "plots": ["png"],  # images/plots
}

# Create folders if they don't exist
for folder in folders:
    folder_path = os.path.join(project_root, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Move files into respective folders
for file in os.listdir(project_root):
    file_path = os.path.join(project_root, file)
    if os.path.isfile(file_path):
        ext = file.split('.')[-1].lower()
        moved = False
        for folder, extensions in folders.items():
            if ext in extensions:
                shutil.move(file_path, os.path.join(project_root, folder, file))
                moved = True
                break
        # Keep other files (like project_structure.txt) in root
        if not moved:
            continue

print("Project files organized successfully!")
