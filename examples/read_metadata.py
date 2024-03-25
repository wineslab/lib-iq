import h5py

def get_metadata_from_mat(path):
    # Carica il file .mat
    with h5py.File(path, 'r') as file:
        info = {}
        for key in file.keys():
            if isinstance(file[key], h5py.Dataset):
                print(f"{key}:")
                print(file[key][:])
                print(f"Tipo di dato: {file[key].dtype}")  # Stampa il tipo di dato
                print()
                info[key] = file[key][:]
        #return info
        return 0

# Esegui la funzione con il percorso del tuo file .mat
#get_metadata_from_mat('/root/libiq-101/examples/iq_samples/2024_01_19_20_48_32_507.mat')
get_metadata_from_mat(str('/root/libiq-101/examples/iq_samples_mat/uav1_6ft_burst1_001.mat'))