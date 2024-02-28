#!/bin/bash

# Controlla se il pacchetto libiq è installato
if pip3 show libiq > /dev/null; then
    # Se il pacchetto è installato, disinstallalo
    echo "Il pacchetto libiq è installato. Disinstallazione in corso..."
    pip3 uninstall libiq -y
else
    echo "Il pacchetto libiq non è installato."
fi

# Esegui il comando di build
python3 -m build

# Controlla se il comando di build è stato eseguito con successo
if [ $? -eq 0 ]; then
    echo "Build completato con successo."

    # Cambia directory
    cd dist

    # Installa il pacchetto libiq
    pip3 install libiq-0.0.1-cp39-cp39-linux_x86_64.whl
else
    echo "Il comando di build non è stato eseguito con successo."
fi