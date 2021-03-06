"""
This script transforms a table of cosmological parameters (the README file 
of AbacusSummit) into a dictionary.

Example
=======
>>> python read_params.py --help
>>> python read_params.py --table README.md

Notes
=====

The table with different parameter values should follow the following format:

                       Table
------------------------------------------------------
Comments: The first line after the comments and list of parameter names refers 
to the first set of cosmological params; the second to the next, etc.
| param1 |  param2 |  param3 |  ...  |  notes  |
| ------ |  ------ |  -----  |  ---  |  ------ | 
|val1-1  |  val1-2 |  val1-3 |  ...  |LCDM base|
|val2-1  |  val2-2 |  val2-3 |  ...  |neutriono|
|  ...   |    ...  |   ...   |  ...  |   ...   |
"""
import numpy as np
import re
import sys
import argparse
import os

stringNames = ['notes','root']
rootParam = 'root'
cosmo_dir = os.path.expanduser('~/repos/AbacusSummit/Cosmologies/')
cosmology_table = os.path.join(cosmo_dir,'README.md')
base_file = os.path.join(cosmo_dir,"abacus_bisection_fast.pre")
sigma8 = ['sigma8_m','sigma8_cb']
h = 'h'
As = 'A_s'
h_def = 0.700001
As_def = 2.00001e-9
args = {'table':cosmology_table}

def main(table):
    # parameter names and row where they can be found
    paramNames, namesRow = list_param_names(table)
    
    # Create a numpy array with all entries in the table and names taken from it
    Params, numCosm = read_table(paramNames,table,namesRow)

    # read all the other parameters
    base_param_dict = read_file(base_file)
    
    indCosm = 3
    param_dict = dict(zip(paramNames,Params[indCosm]))
    print(param_dict)
    #print(paramNames, namesRow, numCosm)

def get_dict(sim_name, table=cosmology_table):
    # parameter names and row where they can be found
    paramNames, namesRow = list_param_names(table)
    
    # Create a numpy array with all entries in the table and names taken from it
    Params, numCosm = read_table(paramNames,table,namesRow)
    
    # given a string such as AbacusSummit_base_c012_ph006, extract 12
    indexCosm = int((sim_name.split('c')[-1]).split('_ph')[0])
    nameCosm = "abacus_cosm%03d"%indexCosm

    # read all the other parameters
    base_param_dict = read_file(base_file)

    param_dict = {}
    for i in range(numCosm):
        
        # TESTING including the base parameters
        for key in base_param_dict.keys():
            try:
                param_dict[key] = float(base_param_dict[key])
            except:
                param_dict[key] = base_param_dict[key]

                
        # parameters from the README.md
        for j, paramName in enumerate(paramNames):
            param_dict[paramName] = Params[i][j]
            
        # return the cosmology selected
        if param_dict[rootParam] == nameCosm:
            return param_dict
    
def list_param_names(fn,output_s8=True):
    # count total number of commented rows
    parRow = 0
    for line in open(fn):
        # if this is the start of the table, exit and parse the parameter names
        if line[:1] == '|':
            paramLine = line
            break
        parRow += 1
    # Assuming that the first line in the table has the parameter names
    try:
        paramLine
    except NameError:
        print("The table should have a line '| par1 | par2 | par3' at the beginning to extract column names!"); exit()
    paramLine = re.sub('^\|\s*', '', paramLine)
    paramLine = re.sub('\s*\|\s*$', '', paramLine)
    paramNames = re.split('\s*\|\s*',paramLine)
    
    assert rootParam in paramNames, "You are either lacking a column with the root name or it is not called "+rootParam
    if output_s8 == False:
        for s8 in sigma8: paramNames.remove(s8)
    return paramNames, parRow

def read_table(paramNames,fn,namesRow):
    # for parameter name row and the horizontal line row
    extraRows = 2

    # This is where the sim numbers start
    startRow = namesRow + extraRows # since next row is |---|...|---|
    numRows = sum([1 for line in open(fn) if line[:1]=='|'])

    # number of parameters and cosmologies
    numParams = len(paramNames)
    numCosm = numRows-extraRows
    paramTypes = np.zeros(numParams,dtype=type)
    for i in range(numParams):
        paramTypes[i] = np.float32 if paramNames[i] not in stringNames else '<U256'
    paramsDt = np.dtype([(paramNames[i],paramTypes[i]) for i in range(numParams)])
    
    # array with all param values for all cosmologies
    Params = np.empty(numCosm,dtype=paramsDt)

    iCosm = 0
    for i,line in enumerate(open(fn)):
        if i < startRow or line[:1] != '|': continue
        # separating the individual row entries in 'line'
        line = re.sub('^\|\s*', '', line)
        line = re.sub('\s*\|\s*$', '', line)
        line = re.split('\s*\|\s*',line)
        for p,parName in enumerate(Params.dtype.names):
            if parName == h and 'TBD' in line[p]: Params[parName][iCosm] = h_def;
            elif parName == As and 'TBD' in line[p]: Params[parName][iCosm] = As_def;
            else: Params[parName][iCosm] = (line[p]);
        iCosm += 1
    return Params, numCosm

def read_file(fn):
    # Parse parameters from file
    names = []
    vals = []
    for line in open(fn):
        # if line empty or commented out, skip
        if line[0] == '#' or line == '\n': continue
        # make sure that things are split name = val1, val2, val3, ... into LHS and RHS
        sp = re.split(r'\s*=\s*', line)
        # We should have 2 things = LHS and RHS or 1 thing if RHS is empty after '='
        assert len(sp) <= 2, \
            "This file cannot be parsed correctly, see the example .ini files"
        # separating into parameter name and value
        n = sp[0]; v = sp[-1]
        if '#' in v: # remove commented out sections
            v = v[:v.find('#')]
        if v == '': continue # remove empty value lines
        v = re.sub('\n','',v)
        vals.append(v)
        names.append(n)
        param_dict = {names[i]:vals[i] for i in range(len(vals))}
    try: # if no neutrinos, classy doesn't like being passed omega_ncdm
        if np.float(param_dict[OmegaNu]) == 0.0: del param_dict[OmegaNu]
    except:
        pass
    return param_dict


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--table', help='table name to read from', default=cosmology_table)
    args = parser.parse_args()
    args = vars(args)
    #print(args['table'])
    exit(main(**args))
