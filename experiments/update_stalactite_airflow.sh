poetry build
scp /home/dmitriy/Projects/vfl-benchmark/dist/stalactite-0.1.0-py3-none-any.whl /home/dmitriy/Projects/vfl-benchmark/experiments/airflow/src/
###
cd /home/dmitriy/Projects/vfl-benchmark/experiments/airflow && docker compose down
cd /home/dmitriy/Projects/vfl-benchmark/experiments/airflow && docker compose build --build-arg username=dmitriy
cd /home/dmitriy/Projects/vfl-benchmark/experiments/airflow && docker compose up -d

### node16
scp /home/dmitriy/Projects/vfl-benchmark/dist/stalactite-0.1.0-py3-none-any.whl dalexandrov@node16.bdcl://nfshome/dalexandrov/SBER/vfl-benchmark-feature-bench/experiments/airflow/src/
scp /home/dmitriy/Projects/vfl-benchmark/experiments/airflow/Dockerfile dalexandrov@node16.bdcl://nfshome/dalexandrov/SBER/vfl-benchmark-feature-bench/experiments/airflow/
scp /home/dmitriy/Projects/vfl-benchmark/experiments/airflow/docker-compose.yaml dalexandrov@node16.bdcl://nfshome/dalexandrov/SBER/vfl-benchmark-feature-bench/experiments/airflow/

ssh -T node16.bdcl << EOF

cd /nfshome/dalexandrov/SBER/vfl-benchmark-feature-bench/experiments/airflow/src
chmod 777 stalactite-0.1.0-py3-none-any.whl
cd ..
docker compose down
docker compose build --build-arg username=dalexandrov
docker compose up -d
EOF

#node15
#ssh -T node15.bdcl << EOF
#
#cd /nfshome/dalexandrov/SBER/vfl-benchmark-feature-bench/experiments/airflow/src
#chmod 777 stalactite-0.1.0-py3-none-any.whl
#cd ..
#docker compose down
#docker compose build --build-arg username=dalexandrov
#docker compose up -d
#EOF

##node14
#ssh -T node14.bdcl << EOF
#
#cd /nfshome/dalexandrov/SBER/vfl-benchmark-feature-bench/experiments/airflow/src
#chmod 777 stalactite-0.1.0-py3-none-any.whl
#cd ..
#docker compose down
#docker compose build --build-arg username=dalexandrov
#docker compose up -d
#EOF

##node13
#ssh -T node13.bdcl << EOF
#
#cd /nfshome/dalexandrov/SBER/vfl-benchmark-feature-bench/experiments/airflow/src
#chmod 777 stalactite-0.1.0-py3-none-any.whl
#cd ..
#docker compose down
#docker compose build --build-arg username=dalexandrov
#docker compose up -d
#EOF