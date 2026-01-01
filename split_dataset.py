"""
Script to split validation data into test and validate sets.
This will move half of the validation videos to the test folder.
"""
import os
import shutil
from pathlib import Path

def split_validation_data():
    """Split validation data into test and validate sets (50/50 split)"""
    valid_path = Path("dataset/valid")
    test_path = Path("dataset/test")
    
    # Create test directories if they don't exist
    test_path.mkdir(parents=True, exist_ok=True)
    (test_path / "fight").mkdir(exist_ok=True)
    (test_path / "noFight").mkdir(exist_ok=True)
    
    # Split fight videos
    fight_valid_path = valid_path / "fight"
    fight_test_path = test_path / "fight"
    
    fight_videos = sorted([f for f in os.listdir(fight_valid_path) if f.endswith('.mp4')])
    split_point = len(fight_videos) // 2
    
    print(f"Splitting {len(fight_videos)} fight videos:")
    print(f"  - Keeping {split_point} in valid")
    print(f"  - Moving {len(fight_videos) - split_point} to test")
    
    for i, video in enumerate(fight_videos):
        src = fight_valid_path / video
        if i >= split_point:
            dst = fight_test_path / video
            shutil.move(str(src), str(dst))
            print(f"  Moved {video} to test")
    
    # Split noFight videos
    noFight_valid_path = valid_path / "noFight"
    noFight_test_path = test_path / "noFight"
    
    noFight_videos = sorted([f for f in os.listdir(noFight_valid_path) if f.endswith('.mp4')])
    split_point = len(noFight_videos) // 2
    
    print(f"\nSplitting {len(noFight_videos)} noFight videos:")
    print(f"  - Keeping {split_point} in valid")
    print(f"  - Moving {len(noFight_videos) - split_point} to test")
    
    for i, video in enumerate(noFight_videos):
        src = noFight_valid_path / video
        if i >= split_point:
            dst = noFight_test_path / video
            shutil.move(str(src), str(dst))
            print(f"  Moved {video} to test")
    
    print("\nDataset split complete!")
    print(f"Train: dataset/train")
    print(f"Valid: dataset/valid")
    print(f"Test:  dataset/test")

if __name__ == "__main__":
    split_validation_data()


