import numpy as np
from sklearn.cluster import DBSCAN


tomatoes = np.array([
    [1.2, 0.5, 1.5, 100, 90, "ripe"],
    [1.3, 0.6, 1.5, 110, 95, "unripe"],
    [2.0, 1.0, 1.8, 120, 100, "ripe"]
])


fx_pixels = 700  
k_density = 0.95  
plant_density = 3  


def calculate_volume(w_pixels, z):
    D_mm = (w_pixels * z) / fx_pixels
    D_cm = D_mm / 10
    return (4/3) * np.pi * (D_cm/2)**3

weights = []
for tomato in tomatoes:
    x, y, z, w, h, ripeness = tomato
    vol = calculate_volume(w, z)
    weight = vol * k_density
    if ripeness == "ripe":
        weights.append(weight)
    elif ripeness == "unripe":
        weights.append(weight * 0.3)  


coords = tomatoes[:, :3]
clustering = DBSCAN(eps=0.3, min_samples=2).fit(coords)
labels = clustering.labels_


unique_labels = set(labels)
yield_per_plant = []
for label in unique_labels:
    if label == -1:  
        yield_per_plant.append(sum(weights[labels == -1]))
    else:
        yield_per_plant.append(sum(weights[labels == label]))

total_yield = sum(yield_per_plant) * plant_density * 2 
print(f"{total_yield / 1000:.2f} ")