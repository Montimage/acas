# Installation

## Prepare environment

```sh
sudo apt update
sudo apt install -y python3-pip
python3 --version
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev
sudo apt install -y python3-venv
mkdir environments
cd environments/
python3 -m venv my_env
ls my_env
source my_env/bin/activate
git clone https://github.com/Montimage/acas.git
```

## Get MMT source code
```sh
git submodule update --init --recursive
git submodule sync
```

## Install MMT

```sh
sudo apt-get install -y libxml2-dev libpcap-dev libconfuse-dev
cd server/
./install-dependencies.sh
```

## Install docker

source: https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04

## Install docker-composer
source: https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-compose-on-ubuntu-20-04

## Install Flask server dependencies
In pyvirtual environment, install server dependencies
```
pip install Flask numpy pandas sklearn tensorflow matplotlib seaborn
```

# READ

- Architecture: D1.7 -> 03-WorkPackages/WP1/D1.7 Reference Architecture
- Advanced Cybersecurity Analysis System (ACAS): D3.5 -> 04-Submitted Deliverables/1st Reporting Period/D3.5. Advanced Cybersecurity Analysis System (ACAS) as a module. It includes the structure of the model, scripts for training and testing the module, script for running the module for the new data (predictions), flask API (described in D3.5) which was the first version (poc) ready for integration, and now for the second version it is Kafka producer that hands out the classification results
- Integration D5.1  (Section 4.7 PUZZLE Security/Edge Analytics and Detection Algorithms) - https://repository.puzzle-h2020.com/nextcloud/index.php/apps/files/?dir=/04-Submitted%20Deliverables/1st%20Reporting%20Period&openfile=12271
