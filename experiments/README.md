# Experiments with VMs

## Yandex Cloud, Sber Cloud, VK Cloud

The script is adjusted to the paths within the virtual machines.
Due to the prerequisites are started on the Yandex Cloud machine (as its ports can be accessed externally),
YC is the master of the VFL experiment

### Launch

1. Change the number of members in an experiment.

- Go to the `configs/config-vm-yc.yml` and set the `common.world_size` parameter to desired number of members
- In `experiments/run-distributed-experiment` change the following variables:

```bash
members_yc=1 # Number of members on Yandex Cloud
members_vk=1 # Number of members on VK Cloud
members_sber=0 # Number of members on Sber Cloud
```

> !**Note**, that sum of the `members_yc + members_vk + members_sber` must be equal to the `common.world_size` parameter
> in config

2. Run the experiment (preliminarily stopping previous launches)

```bash
bash experiments/run-distributed-experiment halt 
bash experiments/run-distributed-experiment run 
```

3. Check the logs of the master container

```bash
bash experiments/run-distributed-experiment master-logs 
```

4. You also can go to `http://<public-yc-ip>:5555/` to check the experiments metrics