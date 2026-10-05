import os
import glob

folder = 'video_to_stitch2'
files = glob.glob(os.path.join(folder, '*.jpg'))

for file in files:
    try:
        os.remove(file)
        print(f"Deleted: {file}")
    except Exception as e:
        print(f"Failed to delete {file}: {e}")