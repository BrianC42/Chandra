'''
Created on Sep 8, 2020

@author: Brian
'''
import subprocess
import os

if __name__ == '__main__':
    print("executing external executable ...")
    '''
    set PYTHONPATH=d:/brian/git/chandra/processes/single
    set PYTHONPATH=%PYTHONPATH%;d:/brian/git/chandra/processes/multiprocess
    set PYTHONPATH=%PYTHONPATH%;d:/brian/git/chandra/processes/child
    set PYTHONPATH=%PYTHONPATH%;d:/brian/git/chandra/utility
    set PYTHONPATH=%PYTHONPATH%;d:/brian/git/chandra/technical_analysis
    set PYTHONPATH=%PYTHONPATH%;d:/brian/git/chandra/td_ameritrade
    set PYTHONPATH=%PYTHONPATH%;d:/brian/git/chandra/machine_learning
    set PYTHONPATH=%PYTHONPATH%;d:/brian/git/chandra/unit_test
    '''
    pPath = "d:/brian/git/chandra/processes/single"
    pPath += ";"
    pPath += "d:/brian/git/chandra/processes/multiprocess"
    pPath += ";"
    pPath += "d:/brian/git/chandra/processes/child"
    pPath += ";"
    pPath += "d:/brian/git/chandra/utility"
    pPath += ";"
    pPath += "d:/brian/git/chandra/technical_analysis"
    pPath += ";"
    pPath += "d:/brian/git/chandra/td_ameritrade"
    pPath += ";"
    pPath += "d:/brian/git/chandra/machine_learning"
    pPath += ";"
    pPath += "d:/brian/git/chandra/unit_test"
    os.environ["PYTHONPATH"] = pPath
    '''
    print(os.environ["PYTHONPATH"])
    print("\npython d:\\brian\\git\\chandra\\unit_test\\external.py")
    subprocess.run("python d:\\brian\\git\\chandra\\unit_test\\external.py")
    print("\npython d:\\brian\\git\\chandra\\unit_test\\external.py -h")
    subprocess.run("python d:\\brian\\git\\chandra\\unit_test\\external.py -h")
    print("\npython d:\\brian\\git\\chandra\\unit_test\\external.py 1 2 3 4")
    subprocess.run("python d:\\brian\\git\\chandra\\unit_test\\external.py 1 2 3 4")
    print("\npython d:\\brian\\git\\chandra\\unit_test\\external.py 1 2 3 4 --sum")
    subprocess.run("python d:\\brian\\git\\chandra\\unit_test\\external.py 1 2 3 4 --sum")
    '''
    cmdstr = "python d:\\brian\\git\\chandra\\unit_test\\external.py"
    pathin = "--pathin p1"
    file1 = "--file f1"
    file2 = "--file f2"
    files = file1 + " " + file2
    field1 = "--field fld1"
    field2 = "--field fld2"
    fields = field1
    pathout = "--pathout p5"
    output = "--output p6"
    subprocess.run(cmdstr + " " +  pathin + " " +  files + " " +  fields + " " +  pathout + " " +  output)
    print("\n... continuing")