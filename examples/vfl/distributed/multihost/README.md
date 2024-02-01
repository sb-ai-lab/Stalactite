# Example with VMs

## Yandex Cloud, Sber Cloud, VK Cloud

The script is adjusted to the paths within the virtual machines.
Due to the prerequisites are started on the Yandex Cloud machine (as its ports can be accessed externally),
YC is the master of the VFL experiment

0. Upload the data to the machines by running (from the root-dir)
```bash
bash examples/utils/upload-files upload
```

### Launch

1. Change the number of members in an experiment.

- Configs for the current experiment are: `examples/configs/logreg-sbol-smm-vm-<yc|vk|sber>.yml` 
- In `experiments/run-distributed-experiment` change the following variables:

```bash
members_yc=0 # Number of members on Yandex Cloud
members_vk=1 # Number of members on VK Cloud
members_sber=1 # Number of members on Sber Cloud
```

> !**Note**, that sum of the `members_yc + members_vk + members_sber` must be equal to the `common.world_size` parameter
> in config

2. Run the experiment (preliminarily stopping previous launches)

```bash
bash examples/vfl/distributed/multihost/logreg_sbol_smm_distributed/run-distributed-experiment halt 
bash examples/vfl/distributed/multihost/logreg_sbol_smm_distributed/run-distributed-experiment run 
```

3. Check the logs of the master container

```bash
bash examples/vfl/distributed/multihost/logreg_sbol_smm_distributed/run-distributed-experiment master-logs 
```

4. You also can go to `http://<public-yc-ip>:5555/` to check the experiments metrics