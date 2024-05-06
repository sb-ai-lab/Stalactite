ssh -t comp_cpu1 'export PATH=\"/home/azureuser/.local/bin:$PATH\"; cd /home/azureuser/data/code/vfl-benchmark && poetry run stalactite master start \
    --detached \
    --config-path examples/configs/test_ag/logreg-sbol-smm_cloud_master.yml'

#ssh -t comp_cpu1 'export PATH=\"/home/azureuser/.local/bin:$PATH\"; cd /home/azureuser/data/code/vfl-benchmark && echo $PATH'