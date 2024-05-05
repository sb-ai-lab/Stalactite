ssh -t comp_cpu1 "cd /home/azureuser/data/code/vfl-benchmark && python3 -m poetry run stalactite master start \
    --detached \
    --config-path examples/configs/test_ag/logreg-sbol-smm_cloud_master.yml"