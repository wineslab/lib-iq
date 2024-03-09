import libiq
import os

path = '/root/iq_samples.bin'

result = libiq.Converter.from_bin_to_sigmf(path)
print(result)

# Verifica se il file esiste prima di chiamare la funzione
#if os.path.isfile(path):
#    result = libiq.from_bin_to_sigmf(path)
#    print("Conversione completata con successo.")
#else:
#    print("Il file specificato non esiste o non è accessibile.")
