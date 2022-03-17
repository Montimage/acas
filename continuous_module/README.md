Obligatory for first time running steps:
1. docker-compose
2. creating topic
3. changing max allowed message size to bigger

## Running docker + kafka
Run: `sudo docker-compose -f continuous_module/docker-compose.yaml up -d`
#If port 2181 is busy with other zookeeper, chceck with:
`sudo lsof -i :2181` or `netstat -ant | grep :2181`
and use kill the pid

To check if both are running `sudo docker ps`

Kafka shell:
`docker exec -it kafka /bin/sh`

Creating topic: 
`kafka-topics.sh --create --zookeeper zookeeper:2181 --replication-factor 1 --partitions 1 --topic messages` 
Listing existing topics `kafka-topics.sh --list --zookeeper zookeeper:2181` 
Info about configuration of topic `kafka-topics.sh --zookeeper zookeeper:2181 --describe --topic messages`

Change retention time for topic:
`kafka-configs.sh --alter --zookeeper zookeeper:2181 --entity-type topics --entity-name messages --add-config retention.ms=60000`

Change max allowed message size (bytes) for topic:
`kafka-configs.sh --alter --zookeeper zookeeper:2181 --entity-type topics --entity-name messages --add-config max.message.bytes=104857600 `


## Running whole script:

File config.config in main project folder keeps info about path used for mmt-probe (input/output), watchdog (folder for monitoring new files from mmt) and paths to model and scaler used for predictions.

1. Running brokers `sudo docker-compose -f continuous_module/docker-compose.yaml up -d`
2. Terminal 1: python3 continuous_module/consumer.py  (it gets all messages at the end)
3. Terminal 2: python3 continuous_module/observer.py (has watchdog and is responsible for monitoring mmt probe creation files)
4. Terminal 3: python3 mmt/mmt-runner.py --> runs mmt probe with an example file

