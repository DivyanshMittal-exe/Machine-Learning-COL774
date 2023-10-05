#!/bin/bash

# Define the source and destination directories
source_dir="./"  # Replace with the actual source directory path
dest_host="divyansh@dkstra.cse.iitd.ac.in"
dest_dir="~/ML/"  # Replace with the actual destination directory path



# Perform rsync operation
rsync -avz --exclude '.git' "$source_dir/" "$dest_host:$dest_dir/"

