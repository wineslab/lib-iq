import zmq

# Crea il contesto zmq
context = zmq.Context()

# Crea il socket zmq
socket = context.socket(zmq.PUSH)
socket.bind("tcp://127.0.0.1:55556")

# Percorso del file
file_path = "/root/demo/iq_samples/iq_sample_captured.bin"

# Dimensione del blocco in byte

# Dimensione del blocco in byte
block_size = 8

try:
    # Apri il file in modalità lettura binaria
    with open(file_path, "rb") as file:
        while True:
            # Leggi un blocco di dati binari
            binary_data = file.read(block_size)
            if len(binary_data) != block_size:
                break
            
            # Se il blocco è vuoto, significa che abbiamo raggiunto la fine del file
            if not binary_data:
                break
            
            # Invia il blocco di dati binari tramite zmq
            socket.send(binary_data)

    print("Invio completato")
        
except FileNotFoundError:
    print(f"Il file {file_path} non è stato trovato.")
except Exception as e:
    print(f"Si è verificato un errore: {str(e)}")

