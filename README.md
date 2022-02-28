### Installing

In order to install and run the service, the following steps need to be followed:
1. Cloning the project
2. Installing third-party programs (MMT-Probe, MMT-SDK, MMT-) by using the script in the project ./server/install-dependencies.sh
3. Running the Python server server.py:
```
python3 server/server.py
```

### Prediction
Prediction currently is trained to detect bot attacks ("Ares (developed by Python): remote shell, file upload/download, capturing" from [CIC IDS database](https://www.unb.ca/cic/datasets/ids-2018.html) )
Performing prediction on existing model (either default one, or changed to a freshly trained one by following the steps in the previous section) can be done by requesting a classification using curl:
```
curl -X POST --data-binary "@anydata.pcap" http://127.0.0.1:5000/classification
```
where anydata.pcap is path to any .pcap file, an exemplary pcap can be found in ./data folder in the project. The request returns ID of the classification that is to be used to retrieve classification using a GET request.
To get the classification result by using curl:
```
curl -X GET http://127.0.0.1:5000/classification/ID_NUM
```
where ID_NUM is the id of the classification returned by POST function that executed classification (above); the request returns the list of features including the predicted classification as a last column malware, where value 0 signifies normal traffic, and value 1 means malicious traffic.
