import numpy as np

data = np.load('C:\\Users\\HP\\OneDrive\\Desktop\\EMG_Prosthetic_Hand\\data\\raw\\participant_2\\close_1778058269.npz')


np.set_printoptions(threshold=np.inf)
for key in data.files:
    print(f"--- Array '{key}' ---")
    print(data[key])
