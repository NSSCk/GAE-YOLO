import numpy as np

def calculate_ripeness_from_size(w_pixels, z_mm, fx_pixels=700, tomato_type="beef"):

    D_mm = (w_pixels * z_mm) / fx_pixels
    
    # （cm³）
    V_cm3 = (4/3) * np.pi * (D_mm / 20)**3  
    
   
    if tomato_type == "beef":
        size_thresholds = {"immature": (0, 50), "ripe": (50, 70), "overripe": (70, np.inf)}
    elif tomato_type == "cherry":
        size_thresholds = {"immature": (0, 25), "ripe": (25, 35), "overripe": (35, np.inf)}
    else:
        raise ValueError("Unsupported tomato type")
    
    if D_mm < size_thresholds["immature"][1]:
        return "immature", D_mm, V_cm3
    elif size_thresholds["ripe"][0] <= D_mm <= size_thresholds["ripe"][1]:
        return "overripe", D_mm, V_cm3
    else:
        return "overripe", D_mm, V_cm3


w_pixels = 120  
z_mm = 1500     
ripeness, diameter, volume = calculate_ripeness_from_size(w_pixels, z_mm)
print(f" {ripeness},  {diameter:.1f}mm,  {volume:.1f}cm³")