'''
Created on Jul 6, 2020

@author: Brian
'''
import configparser
import json
import os
import sys


def get_ini_data(csection):

    try:

        exc_txt = "\nAn exception occurred accessing AppData ini file section"
        config_file = os.getenv('localappdata') + "\\Development\\chandra.ini"
        config = configparser.ConfigParser()
        config.read(config_file)
        config.sections()
        ini_data = config[csection]
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)
    
    return ini_data


def read_config_json(json_file):
    print ("reading configuration details from ", json_file)
    
    json_f = open(json_file, "rb")
    json_config = json.load(json_f)
    json_f.close
    
    return (json_config)


def read_processing_network_json(json_file):
    print ("reading processing network details from ", json_file)
    
    json_f = open(json_file, "rb")
    network_json = json.load(json_f)
    json_f.close
    
    return (network_json)
