import configparser
import subprocess
from pathlib import Path

"""
Automatically runs mmt-probe. Uses ./config.config file for all paths (input pcap, probe directory, output csv path etc)
"""

config_path = './config.config'


def run_mmt(cp):
    """
    Runs mmt-probe as a subprocess, deletes the _0 files (leaves the _1.csv where the actual report is)
    :return:
    """
    config = configparser.ConfigParser()
    config.read(cp)
    mmt_probe_path = config['MMT_PROBE']['mmt_probe_path']
    mmt_conf_path = config['MMT_PROBE']['mmt_conf_path']
    input_pcap_path = config['MMT_PROBE']['input_pcap_path']
    mmt_output_csv = config['MMT_PROBE']['mmt_output_csv']
    print(f'{mmt_conf_path},{input_pcap_path},{mmt_output_csv}')

    if not mmt_conf_path or not input_pcap_path or not mmt_output_csv:
        exit('Config/MMT_PROBE does not contain all needed paths')

    # Run probe on the pcap file
    # Example: ./probe -X input.source="./pcap/3.pcap" -X file-output.output-file="3.csv"
    #           -X file-output.output-dir="./csv/"
    subprocess.call([f"{mmt_probe_path}",
                     "-c", f'{mmt_conf_path}',
                     "-X", f'input.source={input_pcap_path}',
                     "-X", f'file-output.output-dir={mmt_output_csv}'])

    # Deleting _0 files
    for filename in Path(mmt_output_csv).glob(f"*_0.csv"):
        filename.unlink()

    return "MMT finished"


run_mmt(config_path)