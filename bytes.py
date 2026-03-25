import numpy as np

pos_file = 'C:\\Users\\jeong\\GS_Project\\projects\\GaussianExample\\Assets\\GaussianAsset\\0002-point_cloud-iteration_161-point_cloud_pos.bytes'
ids_file = 'C:\\Users\\jeong\\GS_Project\\projects\\GaussianExample\\Assets\\GaussianAsset\\0002-point_cloud-iteration_161-point_cloud_ids.bytes'

# Check POS file
try:
    with open(pos_file, 'rb') as f:
        # Read first 10 Vector3 (10 * 3 * 4 bytes)
        pos_data = np.frombuffer(f.read(10 * 3 * 4), dtype=np.float32).reshape(-1, 3)
        print("--- Unity Pos Bytes (First 10) ---")
        for i, p in enumerate(pos_data):
            print(f"Index {i}: {p}")
except Exception as e:
    print(f"Error reading pos file: {e}")

# Check IDS file
try:
    with open(ids_file, 'rb') as f:
        # Check if it's int32 or uint32
        ids_data = np.frombuffer(f.read(10 * 4), dtype=np.uint32)
        print("\n--- Unity IDs Bytes (First 10) ---")
        print(ids_data)
except Exception as e:
    print(f"Error reading ids file: {e}")