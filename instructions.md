# Pipeline
1. llm_filter_AD
2. Analysis of the filtered out AD result



# Note of running experiments
1. Through submission of slurm script to access GPU
2. Interactive jupyter notebook session.
 - Request access to GPU through terminal
 - Access the GPU, go to folder, do "jupyter notebook --no-browser --port=8888 --ip=$(hostname)"
 - Open a new terminal, connect again to ssh, "ssh -J czhan182@login.rockfish.jhu.edu -L 8889:localhost:8889 czhan182@gpu18" , REPLACE the gpu15 with the gpu you are in
 - Open browser and go to link

 NOTE: make sure the port is not occupied from a previous session!!!
