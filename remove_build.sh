#!/bin/bash

if [ -d "../iq_samples_mat" ]; then
    echo "Rimozione della cartella iq_samples_mat..."
    rm -rf "../iq_samples_mat"
else
    echo "La cartella iq_samples_mat non esiste."
fi


# Rimuovi la cartella dist, se esiste
if [ -d "dist" ]; then
    echo "Rimozione della cartella dist..."
    rm -rf dist
else
    echo "La cartella dist non esiste."
fi

# Rimuovi la cartella libiq.egg-info, se esiste
if [ -d "libiq.egg-info" ]; then
    echo "Rimozione della cartella libiq.egg-info..."
    rm -rf libiq.egg-info
else
    echo "La cartella libiq.egg-info non esiste."
fi

# Controlla se il pacchetto libiq è installato
if pip3 show libiq > /dev/null; then
    # Se il pacchetto è installato, disinstallalo
    echo "Il pacchetto libiq è installato. Disinstallazione in corso..."
    pip3 uninstall libiq -y
else
    echo "Il pacchetto libiq non è installato."
fi