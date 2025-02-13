import os

folder_path = "C:/Machine Learning/Real/Chest_X-Ray/chest_xray/train"
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file == ".DS_Store":
            os.remove(os.path.join(root, file))
            print(f"Deleted: {os.path.join(root, file)}")

folder_path = "C:/Machine Learning/Real/Chest_X-Ray/chest_xray/test"
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file == ".DS_Store":
            os.remove(os.path.join(root, file))
            print(f"Deleted: {os.path.join(root, file)}")

folder_path = "C:/Machine Learning/Real/Chest_X-Ray/chest_xray/val"
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file == ".DS_Store":
            os.remove(os.path.join(root, file))
            print(f"Deleted: {os.path.join(root, file)}")