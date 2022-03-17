### Installing

In order to install and run the service, the following steps need to be followed:
1. Cloning the project
2. While in the project folder executing `git submodule update --init --recursive`
3. Installing third-party programs (MMT-Probe, MMT-SDK, MMT-) by using the script in the project ./server/install-dependencies.sh

### Using Kafka Producer
After installing step:

#### Kafka configuration
Obligatory for running Kafka for the first time:
1. Run: `sudo docker-compose -f continuous_module/docker-compose.yaml up -d`
2. Creating topic in kafka:
Running Kafka shell:
```
docker exec -it kafka /bin/sh
```
In the shell creating topic: 
```
kafka-topics.sh --create --zookeeper zookeeper:2181 --replication-factor 1 --partitions 1 --topic messages
```
Change max allowed message size (bytes) for topic:
```
kafka-configs.sh --alter --zookeeper zookeeper:2181 --entity-type topics --entity-name messages --add-config max.message.bytes=104857600
```


If kafka topic is already created run: `sudo docker-compose -f continuous_module/docker-compose.yaml up -d`

#### Monitoring files created by MMT-probe and making predictions
Configuration including directories of MMT-probe, input files and output path for MMT-probe, ML model and scaler directories can be found in ./config.config

1. Once Kafka configuration is done, run Kafka consumer (to see the messages that will be sent) 
`python3 continuous_module/consumer.py`
2. Running python script
`python3 continuous_module/observer.py`
that runs monitoring of the folder (from config) to which MMT-probe will be saving reports, and on closing a file it is running a ML prediction utilizing the model and scaler from directory provided in config file
3. Running script providing data using MMT-probe
`python3 mmt/mmtRunner.py`
that runs MMT-probe on an exemplary .pcap file.


### Running API
After installing step:

1. Running the Python server server.py:
```
python3 server/server.py
```

### Prediction
Prediction currently is trained to detect bot attacks ("Ares (developed by Python): remote shell, file upload/download, capturing" from [CIC IDS database](https://www.unb.ca/cic/datasets/ids-2018.html) )
Performing prediction on existing model (either default one, or changed to a freshly trained one by following the steps in the previous section) can be done by requesting a classification using curl, for instance in folder ./data/:
```
curl -X POST --data-binary "@short_example.pcapng" http://127.0.0.1:5000/classification
```
where short_example.pcap is path to any .pcap file, an exemplary pcap can be found in ./data folder in the project. The request returns ID of the classification that is to be used to retrieve classification using a GET request.
To get the classification result by using curl:
```
curl -X GET http://127.0.0.1:5000/classification/ID_NUM
```
where ID_NUM is the id of the classification returned by POST function that executed classification (above); the request returns the list of features including the predicted classification as a last column malware, where value 0 signifies normal traffic, and value 1 means malicious traffic.


## Training new model
Train and test datasets need to be in .csv format and contain the features, where last column is the label (normal:0, malicious: 1). Training can be done by using curl, for instance in folder ./data/:
```
curl -X POST "http://127.0.0.1:5000/model?nb_epoch_sae=2&batch_size_sae=16&nb_epoch_cnn=1&batch_size_cnn=32" -F "files=@BotTrain_31704_samples.csv" -F "files=@BotTest_13586_samples.csv"
```
where BotTrain_31704_samples.csv and BotTest_13586_samples.csv are exemplary training and testing datasets. There are couple of additional parameters that can be used for parametrisizing the training:

- batch_size_sae – number of training samples to be utilized before the SAE model internal parameters are updated; it is the same for both SAE models; default value is 16
- nb_epoch_sae – number of complete passes through the training dataset in order to train SAE; it is the same for both SAE models; default value is 10
- batch_size_cnn – number of training samples to be utilized before CNN internal parameters are updated; default value is 32
- nb_epoch_cnn – number of complete passes through the training dataset in order to train CNN; default value is 5

The funciton is saving the model on the server and returing the ID of the model. In order to save the model locally use:
```
curl -X GET http://127.0.0.1:5000/model/ID_NUM --output 'output.h5'
```

where ID_NUM is the model ID returned by previous command, and output.h5 is a name of the file in which the model will be saved.
Getting further results of the training (the results on testing dataset) can be done by using curl:
```
curl -X GET http://127.0.0.1:5000/training-results/1 -F what='conf'
```

Where ID_NUM is the id of the model and the what parameter can be one of three values: conf, preds, stats. The function is returning the .csv file containing the confusion matrix, predictions for test dataset (inputted during training process described before), or statistical metrics (accuracy, recall, precision and F1 score) respectively to the selected option in the what parameter.

### Changing default model for a freshly trained one
In order to change the current used model for performing classification for a new one (trained with commands above) it can be done by execuring the following:
```
curl -X PUT 'http://127.0.0.1:5000/model?id=MODEL_ID'
```

where MODEL_ID is the id of the model returned after succesful training process.


### Update of the mirrored GH repository with updates done in bitbucket repository

On bitbucket (original repo)

```
git remote add upstream git@othergitserver.com:user/mirroredrepo.git
git push upstream --mirror
```

On GH (mirrored repo)

```
git remote update
git pull --ff
```

On Gitlab in acas/

```
git pull origin main
```
