# BEATs Sagemaker

This GitHub repository is made for using [BEATs](https://arxiv.org/abs/2212.09058) on your own dataset and is still a work in progress. In its current form, the repository allow a user to

- Fine-tune BEATs on the [ESC50 dataset](https://github.com/karolpiczak/ESC-50).
- Fine-tune a prototypical network with BEATs as feature extractor on the [ESC50 dataset](https://github.com/karolpiczak/ESC-50).

## Ressources
- BEATs paper: https://arxiv.org/abs/2212.09058
- BEATs official repository: https://github.com/microsoft/unilm/tree/master/beats
- ESC-50 dataset: https://github.com/karoldvl/ESC-50/
- NINAnor repository: https://github.com/NINAnor/rare_species_detections


## Dockerizing

```bash
docker build -t beats -f Dockerfile .
```

**Make sure `ESC-50-master` and `BEATs/BEATs_iter3_plus_AS2M.pt` are stored in your `$DATAPATH` (data folder that is exposed to the Docker container)**

To navigate through the container, run:
```bash
docker run -it \
            -v $PWD:/app \
            -v $DATAPATH:/data \
            beats
```


## Using the software: fine tuning

Providing that `ESC-50-master` and `BEATs/BEATs_iter3_plus_AS2M.pt` are stored in your `$DATAPATH`:

```bash
docker run -v $PWD:/app \
            -v $DATAPATH:/data \
            --gpus all `# if you have GPUs available` \
            beats \
            python3 BEATs_on_ECS50/fine_tune/trainer.py fit --config /app/config.yaml
```

```bash
docker run -v $PWD:/app \
            -v $DATAPATH:/data \
            beats \
            python3 BEATs_on_ECS50/fine_tune/trainer.py fit --config /app/config.yaml
```


## Using the software: training a prototypical network

- Create a miniESC50 dataset in your `$DATAPATH`:

```bash
docker run -v $PWD:/app \
            -v $DATAPATH:/data \
            --gpus all `# if you have GPUs available` \
            beats \
            poetry run data_utils/miniESC50.py
```

- Train the prototypical network:

```bash
docker run -v $PWD:/app \
            -v $DATAPATH:/data \
            --gpus all \
            beats \
            poetry run prototypicalbeats/trainer.py fit --trainer.accelerator gpu --trainer.gpus 1 --data miniESC50DataModule
```
