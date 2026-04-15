import pandas as pd
import numpy as np

# Simulate 1000 machine data points
np.random.seed(42)
data = {
    'Air_Temperature': np.random.normal(300, 2, 1000), # Kelvin
    'Process_Temperature': np.random.normal(310, 2, 1000),
    'Rotational_Speed': np.random.normal(1500, 100, 1000), # RPM
    'Torque': np.random.normal(40, 10, 1000), # Nm
    'Tool_Wear': np.random.randint(0, 250, 1000) # Minutes
}

# Failure Logic: High Torque + High Tool Wear + High Temp = Failure
df = pd.DataFrame(data)
df['Machine_Failure'] = ((df['Torque'] > 60) & (df['Tool_Wear'] > 200) | 
                         (df['Process_Temperature'] > 315)).astype(int)

df.to_csv('machine_data.csv', index=False)
print("Manufacturing dataset 'machine_data.csv' created.")
