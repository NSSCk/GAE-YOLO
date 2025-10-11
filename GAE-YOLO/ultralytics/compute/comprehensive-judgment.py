import numpy as np

def calculate_ripeness(hsv_values, shape_features, diameter_mm, tomato_type="beef"):
  
    weights = {"color": 0.55, "shape": 0.25, "size": 0.20}
    
    # --- 1. ---
    H, S, V = hsv_values
    if tomato_type == "beef":
        H_range = {"immature": (50, 10), "ripe": (10, 25), "overripe": (25, 40)}
    else:  # cherry tomato
        H_range = {"immature": (60, 20), "ripe": (20, 30), "overripe": (30, 50)}
    
    if H >= H_range["immature"][0]:
        S_color = 0.2 * (H - H_range["immature"][0]) / (H_range["immature"][0] - H_range["immature"][1])
    elif H >= H_range["ripe"][0]:
        S_color = 0.2 + 0.6 * (H - H_range["ripe"][0]) / (H_range["ripe"][1] - H_range["ripe"][0])
    else:
        S_color = 0.8 + 0.2 * (40 - S) / 40  
    
    # --- 2. 
    circularity, symmetry = shape_features
    S_shape = 0.6 * circularity + 0.4 * symmetry
    
    # --- 3. 
    if tomato_type == "beef":
        D_ranges = {"immature": 50, "ripe": 70, "overripe": 90}
    else:
        D_ranges = {"immature": 25, "ripe": 35, "overripe": 45}
    
    if diameter_mm < D_ranges["immature"]:
        S_size = 0.3 * diameter_mm / D_ranges["immature"]
    elif diameter_mm <= D_ranges["ripe"]:
        S_size = 0.3 + 0.7 * (diameter_mm - D_ranges["immature"]) / (D_ranges["ripe"] - D_ranges["immature"])
    else:
        S_size = 1.0 - 0.2 * (diameter_mm - D_ranges["ripe"]) / (D_ranges["overripe"] - D_ranges["ripe"])
    
 
    total_score = weights["color"] * S_color + weights["shape"] * S_shape + weights["size"] * S_size
    

    if total_score < 0.4:
        return "immature", total_score
    elif 0.4 <= total_score <= 0.7:
        return "mature", total_score
    else:
        return "overripe", total_score


hsv_values = (15, 80, 70)          
shape_features = (0.85, 0.92)    
diameter_mm = 65                  
result, score = calculate_ripeness(hsv_values, shape_features, diameter_mm)
print(f"{result}, {score:.2f}")