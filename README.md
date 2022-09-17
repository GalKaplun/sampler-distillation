# sampler-distillation
To reproduce the experiments use the provided Dockerfile in `docker/Dockerfile` to build the docker then run the container using `scripts/run_docker.sh`. 
To use the Cifar5m dataset, run: `mkdir .data/; for i in `seq 0 1 9`; do gsutil cp gs://hml-public/datasets/cifar5m/classes/class$i.npy data/;`
