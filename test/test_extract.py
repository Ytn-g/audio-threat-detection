
from capture import extract_from_file

features = extract_from_file("test/test_clips/1-1791-A-26.wav")

print(f"Shape: {features.shape}")     # Should be: (40,)
print(f"Min:   {features.min():.4f}")  # Should be a negative number (MFCCs can be negative)
print(f"Max:   {features.max():.4f}")  # Should be a positive number
print(f"First 5 values: {features[:5].round(4)}")