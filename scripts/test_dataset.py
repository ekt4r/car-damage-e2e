from src.data.dataset import CarDDDataset    

dataset = CarDDDataset(data_dir=None, split="train")
image, target = dataset[0]

print(image.size)
print(target["boxes"].shape)
print(target["boxes"][:3])
print(target["labels"].dtype)