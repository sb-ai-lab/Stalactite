# Distributed multiple host and multiple process examples
##### with VMs: Yandex Cloud, Sber Cloud, VK Cloud

### Multiple host experiment (MH)
The script (`examples/vfl/distributed/multihost/logreg_sbol_smm_multihost`) is adjusted to the paths within the virtual 
machines. Due to the prerequisites are started on the Yandex Cloud machine (as its ports can be accessed externally),
YC is the master of the VFL experiment.
> Paths to the experimental config:
> - Yandex cloud: `examples/configs/logreg-sbol-smm-vm-yc.yml`
> - Sber cloud: `examples/configs/logreg-sbol-smm-vm-sber.yml`
> - VK cloud: `examples/configs/logreg-sbol-smm-vm-vk.yml`

### Single host multiple process (MP) experiment
Although the script (`examples/vfl/distributed/multiprocess/logreg_sbol_smm_multiprocess`) has the same structure to the 
multiple host experiment, it introduces no machine-specific variables.
> Path to the experimental config: `examples/configs/logreg-sbol-smm-multiprocess-vm-yc.yml`


## Experiment launch
0. [_optional for the MH experiment_] Upload the data to the machines by running (from the root-dir)
```bash
bash examples/utils/upload-files upload
```

1. You can change the number of members in an experiment.
- MH experiment:
  1. In `experiments/run-distributed-experiment` change the following variables:
    ```bash
    members_yc=0 # Number of members on Yandex Cloud
    members_vk=1 # Number of members on VK Cloud
    members_sber=1 # Number of members on Sber Cloud
    ```
  2. Adjust `common.world_size` configuration parameter to the sum of the `members_yc + members_vk + members_sber`
    > !**Note**, that sum of the `members_yc + members_vk + members_sber` must be equal to the `common.world_size` 
  > parameter in the config

- MP experiment:
  Just change the `common.world_size` parameter in the configuration file 

2. Stop previous launches by running the `halt` command
```bash
# MH experiment
bash examples/vfl/distributed/multihost/logreg_sbol_smm_multihost halt 
# MP experiment
bash examples/vfl/distributed/multiprocess/logreg_sbol_smm_multiprocess halt 
```

2. Run the experiment
```bash
# MH experiment
bash examples/vfl/distributed/multihost/logreg_sbol_smm_multihost run 
# MP experiment
bash examples/vfl/distributed/multiprocess/logreg_sbol_smm_multiprocess run 
```

3. Check the logs of the master container
```bash
# MH experiment
bash examples/vfl/distributed/multihost/logreg_sbol_smm_multihost master-logs 
# MP experiment
bash examples/vfl/distributed/multiprocess/logreg_sbol_smm_multiprocess master-logs 
```

4. You also can go to `http://<public-yc-ip>:5555/` to check the experiments metrics if the `master.run_mlflow` is set 
to `True` in the config
