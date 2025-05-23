#!/bin/bash

docker build -t libiq -f Dockerfile .

docker run -d -it --name libiq libiq

echo "Creation of the image and run of the container done, get inside with: docker exec -it libiq bash"
