import numpy as np

data = np.load('C:\\Users\\hp\\PycharmProjects\\EMG_Prosthetic_Hand\\data\\features_multi\\filtered\\participant_BT1_features.npz')


np.set_printoptions(threshold=np.inf)
for key in data.files:
    print(f"--- Array '{key}' ---")
    print(data[key])
