'''
Created on Jul 6, 2020

@author: Brian
'''
import os
import configparser
import json

def get_ini_data(csection):
    config_file = os.getenv('localappdata') + "\\Development\\data.ini"
    config = configparser.ConfigParser()
    config.read(config_file)
    config.sections()
    ini_data = config[csection]
    return ini_data

def read_config_json(json_file) :
    print ("reading configuration details from ", json_file)
    
    json_f = open(json_file, "rb")
    json_config = json.load(json_f)
    json_f.close
    
    return (json_config)
