import os
import re
from config import CLOUDY_DIR, CLEAR_DIR

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

def validate_dataset():
    print("--- AkashaLens Dataset Validator ---")
    
    if not os.path.exists(CLOUDY_DIR) or not os.path.exists(CLEAR_DIR):
        print("Error: Dataset directories do not exist.")
        return

    cloudy_images = sorted(os.listdir(CLOUDY_DIR))
    clear_images = sorted(os.listdir(CLEAR_DIR))

    total_cloudy = len(cloudy_images)
    total_clear = len(clear_images)

    print(f"Total Cloudy Images found: {total_cloudy}")
    print(f"Total Clear Images found: {total_clear}")

    if total_cloudy == 0 or total_clear == 0:
        print("Dataset is empty. Add images to the folders.")
        return

    if total_cloudy != total_clear:
        print(f"\nWARNING: Image count mismatch! ({total_cloudy} cloudy vs {total_clear} clear)")

    valid_pairs = []
    
    # Try index-based pairing and show explicit pairings
    pair_count = min(total_cloudy, total_clear)
    print("\nVerifying Image Pairings:")
    for i in range(pair_count):
        c_name = cloudy_images[i]
        cl_name = clear_images[i]
        valid_pairs.append((c_name, cl_name))
        print(f"  [{i+1}] Cloudy: '{c_name}'  <--->  Clear: '{cl_name}'")

    print("\n--- Summary ---")
    print(f"Total Valid Pairs: {len(valid_pairs)}")

    if len(valid_pairs) < 10:
        print("\nNOTE ON DATASET SIZE:")
        print(f"You currently have {len(valid_pairs)} image pairs.")
        print("This is completely fine for testing the end-to-end prototype pipeline (Phase 1).")
        print("The U-Net will overfit, but it allows you to verify that training, inference,")
        print("saving, and frontend UI rendering work properly.")

if __name__ == "__main__":
    validate_dataset()