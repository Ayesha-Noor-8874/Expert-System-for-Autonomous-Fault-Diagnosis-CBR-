import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib as plt

# INITIALIZATION: Loading your specific CSV
# Using the path you provided
FILE_PATH = r"C:\Users\ALLAH\Downloads\industrial_fault_detection_data_1000.csv"
df = pd.read_csv(FILE_PATH)
df['Timestamp']=pd.to_datetime(df['Timestamp'])

# Representation: Knowledge Mapping
Fault_Map={
    0: "Normal Operation (No Fault)",
    1: "Mechanical Vibration / Bearing Issue",
    2: "System Overheating / Thermal Stress"
}

# Retrieve: Finding the most similar past case using Euclidean Distance
def retrieve_similar_case(case_library, current_vibration, current_temp, current_pressure):
    historical_features=case_library[['Vibration (mm/s)','Temperature (°C)','Pressure (bar)']].values
    query_vector= np.array([current_vibration,current_temp,current_pressure])
    distances= np.linalg.norm(historical_features - query_vector, axis=1)
    closest_index=np.argmin(distances)
    return case_library.iloc[closest_index]

# RETAIN: Saving new knowledge back to the CSV file
def retain_new_case(case_library, v, t, p, fault_label):
    new_case={
        'Timestamp': pd.Timestamp.now(),
        'Vibration': v,
        'Temperature': t,
        'Pressure': p,
        'Fault Label': fault_label
    }
    updated_library= pd.concat([case_library, pd.DataFrame([new_case])], ignore_index= True)
    updated_library.to_csv(FILE_PATH, index= False)
    print(f"\n[SYSTEM] New case retained in {FILE_PATH}")
    return updated_library

# Current sensor readings from the machine
v_in, t_in, p_in= 0.85, 115.2, 8.1
# RETRIEVE
past_case= retrieve_similar_case(df, v_in, t_in, p_in)

# REUSE (Displaying the past solution)
diagnosis= Fault_Map.get(int(past_case['Fault Label']),'Unknown')
print(f"-----EXPERT DIAGNOSIS-----")
print(f"Detected Condition: {diagnosis}")
print(f"Based on historical case from: {past_case['Timestamp']}")
df=retain_new_case(df, v_in, t_in, p_in, past_case['Fault Label'])
