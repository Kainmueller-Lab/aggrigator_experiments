import matplotlib.pyplot as plt
from wormbodies_dataset_creation import wormbodies_dataset

wdset = wormbodies_dataset(image_path="/fast/AG_Kainmueller/data/data_wormbodies/test", 
                           mask_path="/fast/AG_Kainmueller/data/data_wormbodies/test",
                           uq_map_path="/fast/AG_Kainmueller/data/UQ_maps/wormbodies/BBBC010_test/fg-bg/dropout/eu",
                           prediction_path="/fast/AG_Kainmueller/data/UQ_maps/wormbodies/BBBC010_test/fg-bg/dropout/pred",
                           semantic_mapping_path="")
                               

sample = wdset[0]

plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.imshow(sample["image"])
plt.title("Image")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(sample["mask"])
plt.title("Mask")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(sample["prediction"])
plt.title("Prediction")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(sample["uq_map"])
plt.title("Uncertainty")
plt.axis("off")

plt.suptitle(f"{sample["sample_name"]}: {wdset.uq_method}")
plt.savefig("test_sample.png")
