import torch
from condition_encoder import ConditionEncoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConditionEncoder().to(device)

x = torch.randn(8, 3, 256, 256).to(device)

condition_map, condition_vector = model(x)

print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Input:", x.shape)
print("Condition map:", condition_map.shape)
print("Condition vector:", condition_vector.shape)

assert condition_map.shape == (8, 256, 16, 16)
assert condition_vector.shape == (8, 128)

print("\nCondition Encoder test: PASSED")