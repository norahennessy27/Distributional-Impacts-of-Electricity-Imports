#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *


# In[3]:


from io import BytesIO, TextIOWrapper
from zipfile import ZipFile
import urllib.request
import csv
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np
from collections import Counter


# In[4]:


# Ensure compatibility between python 2 and python 3
#from __future__ import (absolute_import, division,
 #                       print_function, unicode_literals)
from builtins import *

import requests
import platform
import os
import stat
import tempfile
import json
import time
import subprocess
import geopandas as gpd
import shutil

def _download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = requests.get(url)
        # write to file
        file.write(response.content)

_inmap_exe = None
_tmpdir = tempfile.TemporaryDirectory()


if _inmap_exe == None:
    ost = platform.system()
    print("Downloading InMAP executable for %s"%ost, end='\r')
    if ost == "Windows":
        _inmap_exe = os.path.join(_tmpdir.name, "inmap_1.7.2.exe")
        _download("https://github.com/spatialmodel/inmap/releases/download/v1.7.2/inmap1.7.2windows-amd64.exe", _inmap_exe)
    elif ost == "Darwin":
        _inmap_exe = os.path.join(_tmpdir.name, "inmap_1.7.2")
        _download("https://github.com/spatialmodel/inmap/releases/download/v1.7.2/inmap1.7.2darwin-amd64", _inmap_exe)
    elif ost == "Linux":
        _inmap_exe = os.path.join(_tmpdir.name, "inmap_1.7.2")
        _download("https://github.com/spatialmodel/inmap/releases/download/v1.7.2/inmap1.7.2linux-amd64", _inmap_exe)
    else:
        raise(OSError("invalid operating system %s"%(ost)))
    os.chmod(_inmap_exe, stat.S_IXUSR|stat.S_IRUSR|stat.S_IWUSR)







def run_sr(emis, model, output_variables, emis_units="tons/year"):
    """
    Run the provided emissions through the specified SR matrix, calculating the
    specified output properties.

    Args:
        emis: The emissions to be calculated, Needs to be a geopandas dataframe.

        model: The SR matrix to use. Allowed values:
            isrm: The InMAP SR matrix
            apsca_q0: The APSCA SR matrix, annual average
            apsca_q1: The APSCA SR matrix, Jan-Mar season
            apsca_q2: The APSCA SR matrix, Apr-Jun season
            apsca_q3: The APSCA SR matrix, Jul-Sep season
            apsca_q4: The APSCA SR matrix, Oct-Dec season

        output_variables: Output variables to be calculated. See
            https://inmap.run/docs/results/ for more information.

        emis_units: The units that the emissions are in. Allowed values:
            'tons/year', 'kg/year', 'ug/s', and 'μg/s'.
    """


    global _tmpdir
    global _inmap_exe

    model_paths = {
        "isrm": "/data/isrmv121/isrm_v1.2.1.ncf",
        "apsca_q0": "/data/apsca/apsca_sr_Q0_v1.2.1.ncf",
        "apsca_q1": "/data/apsca/apsca_sr_Q1_v1.2.1.ncf",
        "apsca_q2": "/data/apsca/apsca_sr_Q2_v1.2.1.ncf",
        "apsca_q3": "/data/apsca/apsca_sr_Q3_v1.2.1.ncf",
        "apsca_q4": "/data/apsca/apsca_sr_Q4_v1.2.1.ncf",
    }
    if model not in model_paths.keys():
        models = ', '.join("{!s}".format(k) for (k) in model_paths.keys())
        msg = 'model must be one of \{{!s}\}, but is `{!s}`'.format(models, model)
        raise ValueError(msg)
    model_path = model_paths[model]

    start = time.time()
    job_name = "run_aqm_%s"%start
    emis_file = os.path.join(_tmpdir.name, "%s.shp"%(job_name))
    emis.to_file(emis_file)
    
    try:
        subprocess.check_output([_inmap_exe, "cloud", "start",
            "--cmds=srpredict",
            "--job_name=%s"%job_name,
            "--memory_gb=2",
            "--EmissionUnits=%s"%emis_units,
            "--EmissionsShapefiles=%s"%emis_file,
            "--OutputVariables=%s"%json.dumps(output_variables),
            "--SR.OutputFile=%s"%model_path])
    except subprocess.CalledProcessError as err:
        print(err.output)
        raise

    while True:
        try:
            status = subprocess.check_output([_inmap_exe, "cloud", "status", "--job_name=%s"%job_name]).decode("utf-8").strip()
            print("simulation %s (%.0f seconds)               "%(status, time.time()-start), end='\r')
            if status == "Complete":
                break
            elif status != "Running":
                raise ValueError(status)
        except subprocess.CalledProcessError as err:
            print(err.output)
        time.sleep(10)

    subprocess.check_call([_inmap_exe, "cloud", "output", "--job_name=%s"%job_name])
    output = gpd.read_file("%s/OutputFile.shp"%job_name)

    shutil.rmtree(job_name)
    subprocess.check_call([_inmap_exe, "cloud", "delete", "--job_name=%s"%job_name])

    print("Finished (%.0f seconds)               "%(time.time()-start))

    return output


# In[66]:


#Load data

#Load plant info
plantBAmatch = pd.read_csv("PlantBAmatch2019_new.csv")
plantBAmatch.rename(columns={"Balancing.Authority.Code":"BACODE", "Latitude":"LAT","Longitude":"LON"}, inplace=True)

#Load air pollution data
ap_data_df = pd.read_csv("cems2019.csv")
#print('number of rows in cems data:', len(ap_data_df))
print(ap_data_df.columns)

#Load NEI data
nei_data = pd.read_csv("nei_plant_data_2017_v2.csv")
nei_data.drop(columns = {"CHPFLAG","ELCALLOC","Unnamed: 0"}, inplace = True)
#print('number of rows in nei data:', len(nei_data))

#Load electricity transfer data
electricity_transfer = pd.read_csv("seed_data_elec_2019.csv")
electricity_transfer.drop(electricity_transfer.columns[0],axis = 1, inplace = True)
# electricity_transfer.drop(electricity_transfer.columns[0],axis = 1, inplace = True)
#print(electricity_transfer.head())

electricity_transfer.set_index("X1", inplace=True)

#Merge air pollution data sets
#Note that data from NEI (SO2, NOX, PM25 columns) are in short tons
ap_data_df = pd.merge(ap_data_df, nei_data, on="Plant.Code", how="left")
ap_data_df.drop(columns = ["LAT","LON", "BACODE"], inplace=True)
#CEMS data in metric tons
#ap_data_df.rename(columns={"pm25_per_mwh": "PM25"}, inplace=True)
#print(ap_data_df[ap_data_df.pm25_per_mwh.isna()].head())
#print('number of rows after merge:', len(ap_data_df))

#Load NEI averages by fuel type
#NEI data in short tons
nei_avgs = pd.read_csv("nei_avg_data_2017_v2.csv")
nei_avgs.drop(columns = ['Unnamed: 0'], inplace=True)
nei_avgs.drop([0], inplace=True)
nei_avgs.set_index("PLPRMFL", inplace=True)
nei_avgs.rename(columns ={"PM25": "pm25_per_mwh"}, inplace=True)
nei_avgs.rename(columns ={"VOC": "voc_per_mwh"}, inplace=True)
nei_avgs.rename(columns ={"NH3": "nh3_per_mwh"}, inplace=True)

#Fill in NAs with fuel type averages
for fueltype in nei_avgs.index:
    for col in nei_avgs:
        ap_data_df[col] = np.where(np.logical_and(ap_data_df.PLPRMFL == fueltype, ap_data_df[col].isna()), nei_avgs.loc[fueltype, col], ap_data_df[col])
    

nei_total_avgs = nei_data.agg({"pm25_per_mwh":"mean", "stkhgt":"median", "stkvel":"median","stktemp":"median","stkdiam":"median"})
ap_data_df.loc[ap_data_df["stkhgt"].isna(), "stkhgt"]=nei_total_avgs["stkhgt"]
ap_data_df.loc[ap_data_df["stkvel"].isna(), "stkvel"]=nei_total_avgs["stkvel"]
ap_data_df.loc[ap_data_df["stktemp"].isna(), "stktemp"]=nei_total_avgs["stktemp"]
ap_data_df.loc[ap_data_df["stkdiam"].isna(), "stkdiam"]=nei_total_avgs["stkdiam"]

ap_data_df["PM25"]=ap_data_df["pm25_per_mwh"]*ap_data_df["Total.Gen.MWh"]
ap_data_df["VOC"]=ap_data_df["voc_per_mwh"]*ap_data_df["Total.Gen.MWh"]
ap_data_df["NH3"]=ap_data_df["nh3_per_mwh"]*ap_data_df["Total.Gen.MWh"]



## CORRECT OVEC electricity transfers "generation"
#ovec_ap_data = ap_data_df[ap_data_df['BACODE']=="OVEC"].copy()


#ap_data_df.drop(columns = ['Unnamed: 0_x','PLNGENAN','PLFUELCT','LAT', 'LON', 'PLNOXAN', 'PLSO2AN', 'PLCO2AN','BACODE'], inplace=True)


ovec_ap_data = ap_data_df.loc[ap_data_df['BANAME']=="Ohio Valley Electric Corporation",["OP_DATE_TIME","Total.Gen.MWh"]].copy()
ovec_ap_data = ovec_ap_data.groupby("OP_DATE_TIME").sum()
electricity_transfer["EBA.OVEC-ALL.NG.H"]= ovec_ap_data["Total.Gen.MWh"]




# In[ ]:


#Extract electricity imports and self-generation

def elec_transfers(ba): 
    results = {}
    E = electricity_transfer.filter(regex=ba,axis=1)

    #Exports
    E_export = E.filter(regex=r'EBA.{}'.format(ba))
    #E_export[E_export<0]=0
    E_export = E_export.where(E_export >=0,0) 
    E_export.drop(E_export.filter(regex='ALL').columns, axis=1, inplace = True)

    # Imports
    E_import = E.filter(regex=r'EBA.{}'.format(ba))
    #E_import[E_import>0]=0
    E_import = E_import.where(E_import <0,0)
    E_import.drop(E_import.filter(regex='ALL').columns, axis=1, inplace = True)

    E_self = E.filter(regex=r'EBA.{}.ALL.NG.H'.format(ba), axis=1)
    E_demand = E.filter(regex=r'EBA.{}.ALL.D'.format(ba),axis=1)

    
    if E_demand.empty:
        fBA = 0
    else:
        #Fraction of electricity consumed in BA (as opposed to in exports)
        fBA = E_demand.values[0]/(E_demand.values[0]+E_export.sum(axis=1))
    
    results["imports"]=E_import
    results["generation"]=E_self
    results["all_transfers"]=E
    results["exports"]=E_export
    results["demand"]=E_demand
    results["f_ba"]=fBA
    
    return results


# In[ ]:


#transfers = elec_transfers("CISO")


# In[139]:


#nei_avgs


# In[18]:


#Source: GREET 2013 Update on greenhouse gas and criteria air pollutant emissions factors  (currently using 2019 fleet average ()and 2017 for last 3)
#Emissions factors: Tons/MWh
gas_nox_ef = 0.000139 # 0.0001175 
gas_so2_ef = 0.00000190 # 4.1e-6 
gas_co2_ef = 0.374 #0.441 
gas_pm25_ef = 0.000030 # 9.9208e-7 
gas_nh3_ef = 2.036586e-05 # 0 
gas_voc_ef = 1.206447e-05 # 0 


# In[120]:


def assign_emissions2(import_flag, E_import, E_gen, E_export, f_BA, coal = True, gas = True, coal2gas = False, CCS_coal = False, CCS_gas = False, CCS_coal2gas = False):
    #print(import_flag)
    import_elec = {}
    self_elec = {}
    if import_flag == "import":
            # Generate emissions list for imports
        emissions_output = []
        gen_PP_output = []
        #CO2 = {}
        import_elec = {}

        for col in E_import.columns:
            #Extract name of BA
            gen_BA = col.split('.')[1]
            gen_BA = gen_BA.split('-')[1]
            print(gen_BA)

            #Find all power plants in each BA
            #gen_PP = plantBAmatch[plantBAmatch["BACODE"]==gen_BA].copy()
            gen_PP = plantBAmatch[plantBAmatch["BACODE"]==gen_BA].copy()
            #print(gen_PP)

            #Add up total capacity of generators
            #tot_Capacity = gen_PP["PLNGENAN"].sum()
            #print(gen_BA)
            tot_Capacity = elec_transfers(gen_BA)["generation"]
            #tot_Capacity = CISO_E_gen.values[0]
            #print(tot_Capacity)
            #print(gen_BA)
            #print(tot_Capacity)

            #Calculate fraction of total capacity needed to meet import demand
            #print(tot_Capacity)
            #print(CISO_E_import[col])
            #print(f_BA)
            cap_Frac = -1*E_import[col]*f_BA/(tot_Capacity.squeeze())
            cap_Frac.rename("cap_Frac",inplace=True)
            #print(cap_Frac.head())

            #Assign generation capacity to each plant
            #gen_PP["Export.Capacity.MWh"] = gen_PP["PLNGENAN"]*cap_Frac[0]
            #print(gen_PP.head())
            #print(gen_PP["Export.Capacity.MWh"].sum())

            #Merge plant information with air pollution data
            #print(ap_data_df.head())
            gen_PP_emissions = pd.merge(gen_PP, ap_data_df, on="Plant.Code")
            #gen_PP_emissions.set_index('OP_DATE_TIME',inplace=True)
            #print(gen_PP_emissions.head())
            #print(gen_PP_emissions["Export.Capacity.MWh"].sum())
            #print(gen_PP_emissions.columns)
            gen_PP_emissions = pd.merge(gen_PP_emissions,cap_Frac, right_index=True, left_on = "OP_DATE_TIME", how="left")
            #print(gen_PP_emissions[gen_PP_emissions["OP_DATE_TIME"]== "2016-09-01 02:00:00"])
            
            # DEAL WITH THIS PART: Merge on hour
            gen_PP_emissions["Export.Capacity.MWh"] = gen_PP_emissions["Total.Gen.MWh"]*gen_PP_emissions["cap_Frac"]
            
            gen_PP_out = gen_PP_emissions

            #Assign NOx emissions to power plants
            gen_PP_emissions["Export.NOx.tons"] = gen_PP_emissions["Export.Capacity.MWh"]/gen_PP_emissions["Total.Gen.MWh"]*gen_PP_emissions["Total.NOx.tons"]*1.10231
            #print(gen_PP_emissions.head())

            #Assign SO2 emissions to power plants
            gen_PP_emissions["Export.SO2.tons"] = gen_PP_emissions["Export.Capacity.MWh"]/gen_PP_emissions["Total.Gen.MWh"]*gen_PP_emissions["Total.SO2.tons"]*1.10231
            #print(gen_PP_emissions.head())

            #Assign CO2 emissions to power plants
            gen_PP_emissions["Export.CO2.tons"] = gen_PP_emissions["Export.Capacity.MWh"]/gen_PP_emissions["Total.Gen.MWh"]*gen_PP_emissions["Total.CO2.tons"]*1.10231
            #print(gen_PP_emissions.head())

            #Assign PM2.5 emissions
            gen_PP_emissions["Export.PM25.tons"]= gen_PP_emissions["Export.Capacity.MWh"]/gen_PP_emissions["Total.Gen.MWh"]*gen_PP_emissions["PM25"]

            #Assign NH3 emissions
            gen_PP_emissions["Export.NH3.tons"]= gen_PP_emissions["Export.Capacity.MWh"]/gen_PP_emissions["Total.Gen.MWh"]*gen_PP_emissions["NH3"]

            #Assign VOC emissions
            gen_PP_emissions["Export.VOC.tons"]= gen_PP_emissions["Export.Capacity.MWh"]/gen_PP_emissions["Total.Gen.MWh"]*gen_PP_emissions["VOC"]


            #Assign stack info
            gen_PP_emissions["height"] = gen_PP_emissions["stkhgt"]*0.3048
            gen_PP_emissions["diam"] = gen_PP_emissions["stkdiam"]*0.3048
            gen_PP_emissions["temp"] = (gen_PP_emissions["stktemp"]-32)*5/9+273.15
            gen_PP_emissions["velocity"] = gen_PP_emissions["stkvel"]*0.3048
            
            if coal == False:
                gen_PP_emissions = gen_PP_emissions[gen_PP_emissions.PLFUELCT !="COAL"]
            if gas == False:
                gen_PP_emissions = gen_PP_emissions[gen_PP_emissions.PLFUELCT !="GAS"]
            if coal2gas == True:
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL", "Export.NOx.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_nox_ef
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.SO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_so2_ef
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.CO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_co2_ef
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.PM25.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_pm25_ef
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.NH3.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_nh3_ef
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.VOC.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_voc_ef
            if CCS_coal == True:
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.NOx.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.NOx.tons"]*1.2 #20% energy penalty
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.SO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.SO2.tons"]*1.2*0 #100% capture
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.CO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.CO2.tons"]*1.2*0.1 #90% capture
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.PM25.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.PM25.tons"]*1.2*0.3 #70% capture
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.NH3.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*0.23 * 0.00110231 #kg to short tons
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.VOC.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.VOC.tons"]*1.2 #scales with energy penalty
            if CCS_gas == True:
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.NOx.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.NOx.tons"]*1.15 #15% energy penalty
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.SO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.SO2.tons"]*1.15*0 #100% capture
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.CO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.CO2.tons"]*1.15*0.1 #90% capture
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.PM25.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.PM25.tons"]*1.15 #scales with energy penalty
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.NH3.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.Capacity.MWh"]*0.002 * 0.00110231
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.VOC.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.VOC.tons"]*1.15 #scales with energy penalty
            if CCS_coal2gas == True:
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL", "Export.NOx.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_nox_ef*1.15
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.SO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_so2_ef*1.15*0
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.CO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_co2_ef*1.15*0.1
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.PM25.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_pm25_ef*1.15
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.NH3.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*0.002*0.00110231
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.VOC.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_voc_ef*1.15
                
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.NOx.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.NOx.tons"]*1.15
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.SO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.SO2.tons"]*1.15*0
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.CO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.CO2.tons"]*1.15*0.1
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.PM25.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.PM25.tons"]*1.15
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.NH3.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.Capacity.MWh"]*0.002*0.00110231 # kg to short tons
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.VOC.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.VOC.tons"]*1.15
           


                
                
            #print(gen_PP_emissions.head())

            emissions_output.append(gen_PP_emissions[["Export.SO2.tons","Export.NOx.tons","Export.PM25.tons","Export.NH3.tons","Export.VOC.tons","height","diam","temp","velocity","LAT","LON", "Export.CO2.tons", "Export.Capacity.MWh","Plant.Code"]])
            #emissions_output.append(gen_PP_emissions[["Export.SO2.tons","Export.NOx.tons","Latitude","Longitude"]])#,"Export.PM25.tons","height","diam","temp","velocity","LAT","LON"]])
            #print(emissions_output)
            #print('made it to emissions output')

            #CO2[gen_BA] = gen_PP_emissions["Export.CO2.tons"].sum()
            import_elec[gen_BA] = gen_PP_emissions["Export.Capacity.MWh"].sum()
            
            #gen_PP_output.append(gen_PP[["BACODE","Export.Capacity.MWh", "PLFUELCT","LAT","LON"]])
            gen_PP_output.append(gen_PP_out[["BACODE","Export.Capacity.MWh","LAT","LON"]])
            #print(gen_PP_output)

        #emissions_output
        emissions = pd.concat(emissions_output)
        #print(emissions)
        generation = pd.concat(gen_PP_output)
    elif import_flag == "self":
        # Generate emissions list for imports
        emissions_output = []
        gen_PP_output = []
        self_elec = {}

        for col in E_gen.columns:
            #Extract name of BA
            gen_BA = col.split('.')[1]
            gen_BA = gen_BA.split('-')[0]
            #print(gen_BA)

            #Find all power plants in each BA
            #gen_PP = plantBAmatch[plantBAmatch["BACODE"]==gen_BA].copy()
            gen_PP = plantBAmatch[plantBAmatch["BACODE"]==gen_BA].copy()
            #print(gen_PP)

            #Add up total capacity of generators
            #tot_Capacity = gen_PP["PLNGENAN"].sum()
            #print(gen_BA)
            tot_Capacity = E_gen
            #print(tot_Capacity)
            
            self_gen = E_gen[col]*f_BA

            #Calculate fraction of total capacity needed to meet import demand
            cap_Frac = self_gen/(tot_Capacity.squeeze())
            cap_Frac.rename("cap_Frac",inplace=True)
            #print(cap_Frac)

            #Assign generation capacity to each plant
            #gen_PP["Export.Capacity.MWh"] = gen_PP["PLNGENAN"]*cap_Frac[0]
            #print(gen_PP.head())

            #Merge plant information with air pollution data
            gen_PP_emissions = pd.merge(gen_PP, ap_data_df, on="Plant.Code")
            #print(gen_PP_emissions)
            #print(gen_PP_emissions["Export.Capacity.MWh"].sum())
            gen_PP_emissions = pd.merge(gen_PP_emissions,cap_Frac, right_index=True, left_on = "OP_DATE_TIME", how="left")
            
            gen_PP_emissions["Export.Capacity.MWh"] = gen_PP_emissions["Total.Gen.MWh"]*gen_PP_emissions["cap_Frac"]
            
            gen_PP_out = gen_PP_emissions

            #Assign NOx emissions to power plants
            gen_PP_emissions["Export.NOx.tons"] = gen_PP_emissions["Export.Capacity.MWh"]/gen_PP_emissions["Total.Gen.MWh"]*gen_PP_emissions["Total.NOx.tons"]*1.10231
            #print(gen_PP_emissions.head())

            #Assign CO2 emissions to power plants
            gen_PP_emissions["Export.CO2.tons"] = gen_PP_emissions["Export.Capacity.MWh"]/gen_PP_emissions["Total.Gen.MWh"]*gen_PP_emissions["Total.CO2.tons"]*1.10231
            #print(gen_PP_emissions.head())

            #Assign SO2 emissions to power plants
            gen_PP_emissions["Export.SO2.tons"] = gen_PP_emissions["Export.Capacity.MWh"]/gen_PP_emissions["Total.Gen.MWh"]*gen_PP_emissions["Total.SO2.tons"]*1.10231
            #print(gen_PP_emissions.head())

            #Assign PM2.5 emissions
            gen_PP_emissions["Export.PM25.tons"]= gen_PP_emissions["Export.Capacity.MWh"]/gen_PP_emissions["Total.Gen.MWh"]*gen_PP_emissions["PM25"]

            #Assign NH3 emissions
            gen_PP_emissions["Export.NH3.tons"]= gen_PP_emissions["Export.Capacity.MWh"]/gen_PP_emissions["Total.Gen.MWh"]*gen_PP_emissions["NH3"]

            #Assign VOC emissions
            gen_PP_emissions["Export.VOC.tons"]= gen_PP_emissions["Export.Capacity.MWh"]/gen_PP_emissions["Total.Gen.MWh"]*gen_PP_emissions["VOC"]


            #Assign stack info
            gen_PP_emissions["height"] = gen_PP_emissions["stkhgt"]*0.3048
            gen_PP_emissions["diam"] = gen_PP_emissions["stkdiam"]*0.3048
            gen_PP_emissions["temp"] = (gen_PP_emissions["stktemp"]-32)*5/9+273.15
            gen_PP_emissions["velocity"] = gen_PP_emissions["stkvel"]*0.3048
            
            if coal == False:
                gen_PP_emissions = gen_PP_emissions[gen_PP_emissions.PLFUELCT !="COAL"]
            if gas == False:
                gen_PP_emissions = gen_PP_emissions[gen_PP_emissions.PLFUELCT !="GAS"]
            if coal2gas == True:
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL", "Export.NOx.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_nox_ef
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.SO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_so2_ef
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.CO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_co2_ef
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.PM25.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_pm25_ef
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.NH3.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_nh3_ef
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.VOC.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_voc_ef
            if CCS_coal == True:
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.NOx.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.NOx.tons"]*1.2
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.SO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.SO2.tons"]*1.2*0
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.CO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.CO2.tons"]*1.2*0.1
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.PM25.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.PM25.tons"]*1.2*0.3
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.NH3.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*0.23*0.00110231 #kg to short tons
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.VOC.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.VOC.tons"]*1.2
            if CCS_gas == True:
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.NOx.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.NOx.tons"]*1.15
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.SO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.SO2.tons"]*1.15*0
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.CO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.CO2.tons"]*1.15*0.1
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.PM25.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.PM25.tons"]*1.15
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.NH3.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.Capacity.MWh"]*0.002*0.00110231 # kg to short tons
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.VOC.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.VOC.tons"]*1.15

            if CCS_coal2gas == True:
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL", "Export.NOx.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_nox_ef*1.15
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.SO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_so2_ef*1.15*0
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.CO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_co2_ef*1.15*0.1
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.PM25.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_pm25_ef*1.15
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.NH3.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*0.002*0.00110231
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "COAL","Export.VOC.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "COAL"]["Export.Capacity.MWh"]*gas_voc_ef*1.15
                
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.NOx.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.NOx.tons"]*1.15
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.SO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.SO2.tons"]*1.15*0
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.CO2.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.CO2.tons"]*1.15*0.1
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.PM25.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.PM25.tons"]*1.15
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.NH3.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.Capacity.MWh"]*0.002*0.00110231 # kg to short tons
                gen_PP_emissions.loc[gen_PP_emissions.PLFUELCT == "GAS","Export.VOC.tons"] = gen_PP_emissions[gen_PP_emissions.PLFUELCT == "GAS"]["Export.VOC.tons"]*1.15
           


            emissions_output.append(gen_PP_emissions[["Export.SO2.tons","Export.NOx.tons","Export.PM25.tons","Export.NH3.tons","Export.VOC.tons","height","diam","temp","velocity","LAT","LON", "Export.CO2.tons","Export.Capacity.MWh","Plant.Code"]])
            #emissions_output.append(gen_PP_emissions[["Export.SO2.tons","Export.NOx.tons","Latitude","Longitude"]])#,"Export.PM25.tons","height","diam","temp","velocity","LAT","LON"]])

            #print(emissions_output)
            self_elec[gen_BA] = gen_PP_emissions["Export.Capacity.MWh"].sum()
            
            #gen_PP_output.append(gen_PP[["BACODE","Export.Capacity.MWh", "PLFUELCT","LAT","LON"]])
            gen_PP_output.append(gen_PP_out[["BACODE","Export.Capacity.MWh","LAT","LON"]])
            #print(gen_PP_output)

        #emissions_output
        emissions = pd.concat(emissions_output)
        #print(emissions)
        
        generation = pd.concat(gen_PP_output)
        #print(generation)
        
    
    return emissions, generation


# In[ ]:


#CISO = assign_emissions2("self", transfers["imports"], transfers["generation"], transfers["exports"], 
#                         transfers["f_ba"], coal=True, gas=True, CCS_gas = False, CCS_coal = False)

def get_emissions_all(BA_list,import_flag, coal, gas, coal2gas, CCS_coal, CCS_gas, CCS_coal2gas): 
    emissions_output = []
    generation_output = []
    for ba in BA_list:
        transfers = elec_transfers(ba)
        try:
            emissions, generation =  assign_emissions2(import_flag, transfers["imports"], transfers["generation"], 
                                        transfers["exports"], transfers["f_ba"], coal, gas, 
                                           coal2gas, CCS_coal, CCS_gas, CCS_coal2gas)
        except ValueError as e:
            print(e)
            continue
        emissions_output.append(emissions)
        generation_output.append(generation)
    
    return pd.concat(emissions_output), pd.concat(generation_output)


# In[ ]:


#Calculate emissions totals a BA has responsibility for
#Input: result of assign_emissions function: emissions and generation from each power plant
#Output: total emissions the BA is responsible for through self-generation or imports
def sum_emissions(BA_emissions):
    Tot_SO2 = BA_emissions["Export.SO2.tons"].sum()
    Tot_NOx = BA_emissions["Export.NOx.tons"].sum()
    Tot_CO2 = BA_emissions["Export.CO2.tons"].sum()
    Tot_PM25 = BA_emissions["Export.PM25.tons"].sum()
    Tot_NH3 = BA_emissions["Export.NH3.tons"].sum()
    Tot_VOC = BA_emissions["Export.VOC.tons"].sum()
    return Tot_SO2, Tot_NOx, Tot_CO2, Tot_PM25, Tot_NH3, Tot_VOC

def sum_elec(BA_generation):
    Tot_elec = BA_generation["Export.Capacity.MWh"].sum()
    return Tot_elec
 


# In[ ]:


def create_emissions_file(BA_emissions):
    #emissions = pd.read_csv("emissions_fileCISO_self.csv")
    BA_emissions.rename(columns={"Export.SO2.tons":"SOx", "Export.NOx.tons":"NOx", "Export.PM25.tons":"PM2_5", "Export.NH3.tons":"NH3","Export.VOC.tons":"VOC"}, inplace=True)
    BA_emissions.drop(['Export.CO2.tons','Export.Capacity.MWh'], axis=1, inplace=True)
    #emissions.rename(columns={"Export.SO2.tons":"SOx", "Export.NOx.tons":"NOx"},inplace=True) #"Export.PM25.tons":"PM2_5"}, inplace=True)
    #print('renamed columns')
    #BA_emissions["VOC"]=0
    #BA_emissions["NH3"]=0
    #emissions["PM2_5"]=0
    #emissions["height"]=0
    #emissions["diam"]=0
    #emissions["temp"]=0
    #emissions["velocity"]=0
    #print("got emissions")  
    BA_emissions = BA_emissions.groupby("Plant.Code").agg({"SOx":"sum","NOx":"sum","PM2_5":"sum","height":"mean","diam":"mean",
                                            "temp":"mean","velocity":"mean","VOC":"sum","NH3":"sum", "LAT": "mean","LON": "mean"})
    
    geo_emissions = gpd.GeoDataFrame(
    BA_emissions.drop(['LAT','LON'],axis=1),
    crs={'init': 'epsg:4326'},
    #geometry=[Point(xy) for xy in zip(emissions.Longitude, emissions.Latitude)])
    geometry=[Point(xy) for xy in zip(BA_emissions.LON, BA_emissions.LAT)])
    print('got geometry')
    emis=geo_emissions
    emis.replace([np.inf,-np.inf],np.nan, inplace=True)
    emis.dropna(inplace = True)
    
    return emis


# In[ ]:


#test_emissions = create_emissions_file(CISO[0].copy())


# In[121]:


#model choice = 'isrm', "apsca_q0"
def run_model(model_choice, emissions_file):
    print(model_choice)
    output_variables = {
        'TotalPM25':'PrimaryPM25 + pNH4 + pSO4 + pNO3 + SOA',
        'deathsK':'(exp(log(1.06)/10 * TotalPM25) - 1) * TotalPop * 1.06115917 * MortalityRate / 100000 * 1.036144578',
        'deathsL':'(exp(log(1.14)/10 * TotalPM25) - 1) * TotalPop * 1.06115917 * MortalityRate / 100000 * 1.036144578',
        'Population': 'TotalPop * 1.06115917',
        'Mortality': 'MortalityRate * 1.036144578'
    }
    
    resultsISRM = run_sr(emissions_file, model=model_choice, emis_units="tons/year", output_variables=output_variables)
    return resultsISRM


# In[ ]:


#test_results = run_model("isrm",test_emissions)


# In[54]:


#test_results["deathsL"].sum()


# In[55]:


#sum_emissions(CISO[0])


# In[122]:


#level = "BA","county","block group"
#CR = "K","L"
def aggregate_results(results_raw, level, CR):
    
    #Define all BAs
    BA = pd.read_csv("ba_tz.csv")
    BA["BANAME"]=BA["BANAME"].str.upper()
    #print(BA)
    
    
    if level == "BA":
        #Read in map of control areas
        Control_areas = gpd.read_file("Control_Areas.shp")
        Control_areas.crs

        #Project to match Control Areas crs
        results_raw.to_crs(crs=Control_areas.crs, inplace = True)
        results_raw.crs
        
        results = gpd.sjoin(Control_areas,results_raw,how="inner")
        count = Counter(results["index_right"])
        #print(count)
        for i in range(len(results)):
            n = count[results["index_right"].iloc[i]]
            #n = len(results[results["index_right"]==results["index_right"].iloc[i]])
            #print(n)
            if n>1:
                #print(results["Population"].iloc[i])
                results["Population"].iloc[i] = results["Population"].iloc[i]/n
                #print(results["Population"].iloc[i])
                results["deathsK"].iloc[i] = results["deathsK"].iloc[i]/n
                results["deathsL"].iloc[i] = results["deathsL"].iloc[i]/n
        #print(results)
        
        if CR == "K":
            deaths = results.groupby(["NAME"]).sum()['deathsK']
        elif CR == "L":
            deaths = results.groupby(["NAME"]).sum()['deathsL']
        deaths = pd.DataFrame({'Name':deaths.index,'Deaths':deaths.values})
        population = results.groupby(["NAME"]).sum()['Population']
    
        deaths_aggregated = pd.merge(deaths,BA,how="left", left_on = "Name",right_on = "BANAME")
        #print(deaths_aggregated)
        deaths_aggregated = pd.merge(deaths_aggregated, population, how="left",left_on="Name",right_on = "NAME")
        
        
    elif level == "county":
        #Read in map of counties
        Counties = gpd.read_file("tl_2016_us_county.shp")
        #Counties.to_crs(crs=Control_areas.crs, inplace=True)
        results_raw.to_crs(crs=Counties.crs, inplace = True)
        
        results = gpd.sjoin(Counties, results_raw, how="inner")
        
        count = Counter(results["index_right"])
        for i in range(len(results)):
            n = count[results["index_right"].iloc[i]]
            if n>1:
                results["Population"].iloc[i] = results["Population"].iloc[i]/n
                results["deathsK"].iloc[i] = results["deathsK"].iloc[i]/n
                results["deathsL"].iloc[i] = results["deathsL"].iloc[i]/n
        
        if CR == "K":
            deaths = results.groupby(["STATEFP","NAME","GEOID"]).sum()['deathsK']
            deaths = deaths.reset_index()
            deaths.rename(columns = {"deathsK":"Deaths"}, inplace = True)
        elif CR == "L":
            deaths = results.groupby(["STATEFP","NAME","GEOID"]).sum()['deathsL']  
            deaths = deaths.reset_index()
            deaths.rename(columns = {"deathsL":"Deaths"}, inplace = True)
        #print(deaths.head())
        population = results.groupby(["STATEFP","NAME","GEOID"]).sum()['Population']
        
        deaths_aggregated = pd.merge(deaths, population, how="left",left_on=["STATEFP","NAME","GEOID"],right_on=["STATEFP","NAME","GEOID"])
        
    deaths_aggregated['Deaths_pc'] = deaths_aggregated["Deaths"]/deaths_aggregated["Population"]*100000
    
    
    return deaths_aggregated




# In[81]:


#BA_results = aggregate_results(test_results, 'county',"K")


# In[82]:


#BA_results[BA_results.Deaths > 1]


# In[123]:


def count_deaths(ba, import_flag, model_choice, level, CR, coal_flag, gas_flag, coal2gas, CCS_coal, CCS_gas, CCS_coal2gas ):
    print(ba, import_flag, model_choice, level, CR)
    #Identify electricity transfers for given BA
    transfers = elec_transfers(ba)
    #Assign emissions to each plant
    emissions_plant, generation_plant = assign_emissions2(import_flag, transfers["imports"], transfers["generation"], transfers["exports"], transfers["f_ba"], coal_flag, gas_flag, coal2gas, CCS_coal, CCS_gas,CCS_coal2gas)
    #Track emissions and generation
    if import_flag == "import":
        generation = transfers["imports"].sum(axis=1)*-1*transfers["f_ba"]
    elif import_flag == "self":
        generation = transfers["generation"].sum(axis=1)*transfers["f_ba"]
     
    emissions = sum_emissions(emissions_plant)
    #generation = sum_elec(generation_plant)
    #process emissions file
    emissions_file = create_emissions_file(emissions_plant)
    #run model
    raw_results = run_model(model_choice, emissions_file)
    
    output_K = {}
    output_L = {}
    if "K" in CR:
    
        if level == "BA":
            #aggregate results
            results_BA_K = aggregate_results(raw_results, "BA", "K")
            results_county_K=[]
        elif level == "county":
            results_BA_K = []
            results_county_K = aggregate_results(raw_results, "county", "K")
        elif level == "all":
            results_BA_K = aggregate_results(raw_results, "BA", "K")
            results_county_K = aggregate_results(raw_results, "county", "K")

        output_K["results_BA"] = results_BA_K
        output_K["results_county"] = results_county_K
        output_K["emissions"] = emissions
        output_K["generation"] = generation
        output_K["plants"] = emissions_file
        #output["raw"] = raw_results_K
        
    if "L" in CR:      
        if level == "BA":
            #aggregate results
            results_BA_L = aggregate_results(raw_results, "BA", "L")
            results_county_L=[]
        elif level == "county":
            results_BA_L = []
            results_county_L = aggregate_results(raw_results, "county", "L")
        elif level == "all":
            results_BA_L = aggregate_results(raw_results, "BA", "L")
            results_county_L = aggregate_results(raw_results, "county", "L")

        output_L["results_BA"] = results_BA_L
        output_L["results_county"] = results_county_L
        output_L["emissions"] = emissions
        output_L["generation"] = generation
        output_L["plants"] = emissions_file
        #output["raw"] = raw_results_L

        
        
    return output_K, output_L
   


# In[104]:


#test_run_self_K, test_run_self_L = count_deaths("CISO","self","isrm","BA","KL",gas = True,coal = True, coal2gas = False,CCS_coal = False,CCS_gas = False)


def count_deaths_multi(BA_list, import_flag, model_choice, level, CR, coal_flag, gas_flag, coal2gas, CCS_coal, CCS_gas,CCS_coal2gas):
    emissions_plant, generation_plant = get_emissions_all(BA_list,import_flag, coal_flag, gas_flag, coal2gas, CCS_coal, CCS_gas,CCS_coal2gas)
    
    #Track emissions and generation
    if import_flag == "import":
        #generation = transfers["imports"].sum(axis=1)*-1*transfers["f_ba"]
        generation = []
    elif import_flag == "self":
        #generation = transfers["generation"].sum(axis=1)*transfers["f_ba"]
        generation = []
    
    emissions = sum_emissions(emissions_plant)
    #generation = sum_elec(generation_plant)
    #process emissions file
    emissions_file = create_emissions_file(emissions_plant)
    #run model
    raw_results = run_model(model_choice, emissions_file)
    
    output_K = {}
    output_L = {}
    if "K" in CR:
    
        if level == "BA":
            #aggregate results
            results_BA_K = aggregate_results(raw_results, "BA", "K")
            results_county_K=[]
        elif level == "county":
            results_BA_K = []
            results_county_K = aggregate_results(raw_results, "county", "K")
        elif level == "all":
            results_BA_K = aggregate_results(raw_results, "BA", "K")
            results_county_K = aggregate_results(raw_results, "county", "K")

        output_K["results_BA"] = results_BA_K
        output_K["results_county"] = results_county_K
        output_K["emissions"] = emissions
        output_K["generation"] = generation
        output_K["plants"] = emissions_file
        output_K["raw"] = raw_results
        
    if "L" in CR:      
        if level == "BA":
            #aggregate results
            results_BA_L = aggregate_results(raw_results, "BA", "L")
            results_county_L=[]
        elif level == "county":
            results_BA_L = []
            results_county_L = aggregate_results(raw_results, "county", "L")
        elif level == "all":
            results_BA_L = aggregate_results(raw_results, "BA", "L")
            results_county_L = aggregate_results(raw_results, "county", "L")

        output_L["results_BA"] = results_BA_L
        output_L["results_county"] = results_county_L
        output_L["emissions"] = emissions
        output_L["generation"] = generation
        output_L["plants"] = emissions_file
        output_L["raw"] = raw_results

        
        
    return output_K, output_L

# In[107]:


#test_run_self_L["results_BA"]["Deaths"].sum()


# In[108]:


#####Make list of sorted BAs
def sort_BAs():

    Control_areas = gpd.read_file("Control_Areas.shp")
    centroids = Control_areas.centroid
    Control_areas["centroids"] = centroids
    Control_areas.head()


    Control_areas["latitude"] = Control_areas.centroids.apply(lambda c: c.y)
    Control_areas["longitude"] = Control_areas.centroids.apply(lambda c: c.x)
    #Control_areas.length
    Control_areas.head()
    BAs_sort = pd.DataFrame(Control_areas.loc[:,["NAME","latitude","longitude"]])
    BAs_sort.head()
    BAs_sort.rename(columns={"NAME":"BANAME"}, inplace=True)

    BA = pd.read_csv("ba_tz.csv")
    BA["BANAME"]=BA["BANAME"].str.upper()
    BA.head()

    BAs_sort = pd.merge(BAs_sort, BA, on="BANAME")
    BAs_sort.sort_values(by=["longitude","latitude"], inplace=True)
    BAs_sort.drop_duplicates(subset="BACODE",inplace=True)

    return BAs_sort


# In[124]:


from concurrent.futures import ThreadPoolExecutor

def count_deaths_all_bas(run_codes, import_flag, model_choice, level, CR, coal_flag, gas_flag, coal2gas, CCS_coal, CCS_gas, CCS_coal2gas,results):
    start_time_imports = pd.datetime.now()
    #BA = pd.read_csv("ba_tz.csv")
    #codes = BA["BACODE"]

    #codes = ["CISO","BPAT"]

    #subprocess.CalledProcessError
    def ba_function(ba,import_flag, model_choice, level, CR, coal_flag, gas_flag, coal2gas, CCS_coal, CCS_gas,CCS_coal2gas,result ):
        if result is not None:
            return ba, result
        print(ba)
        attempt = 1
        while True:
            try:
                result = count_deaths(ba,import_flag, model_choice, level, CR, coal_flag, gas_flag, coal2gas, CCS_coal, CCS_gas,CCS_coal2gas )
            except subprocess.CalledProcessError as e:
                if attempt < 1:
                    attempt +=1
                    continue
                result = e
            except Exception as e:
                result = e
            break

        return ba, result

    with ThreadPoolExecutor(max_workers=6) as executor:
        #results=list(executor.map(ba_function,codes,["import"],["isrm"],["BA"],["L"]))
        results = list(executor.map(lambda ba: ba_function(ba,import_flag,model_choice,level,CR, coal_flag, gas_flag, coal2gas, CCS_coal, CCS_gas,CCS_coal2gas,results[ba] if ba in results else None),run_codes))

    output = {k: v for k,v in results if not isinstance(v,Exception)}
    errors = {k: v for k,v in results if isinstance(v,Exception)}

    end_time_imports = pd.datetime.now()
    return output, errors


# In[110]:



#Create SR Matrices
#BAs_sorted = sorted list of BA codes
#row_BAs: the list of BAs causing damaged (those the analysis is done for)
#affected_BAs: the list of all BAs where impacts are felt
def create_matrices(BAs_sort, deaths, responsible_BAs, affected_BAs, model_choice, CR, csv_flag):
    #rows = importing_codes = BA causing damage
    #columns = codes = BA where deaths occur

    affected_BAs = set(affected_BAs)
    codes = [BA for BA in BAs_sort["BACODE"] if BA in affected_BAs]
    #print("codes")
    #print(codes)
    
    matrices = {}

    importing_codes = [BA for BA in BAs_sort["BACODE"] if BA in deaths]

    deaths_df = pd.DataFrame(index=importing_codes,columns=codes)
    #print(deaths_df.columns)
    deaths_df_pc = pd.DataFrame(index=importing_codes,columns=codes)
    
   
    for ba_row in importing_codes:
        for ba_col in codes:
            deaths_df.loc[ba_row, ba_col] = deaths[ba_row]["results_BA"][deaths[ba_row]["results_BA"]["BACODE"]==ba_col]["Deaths"].values[0]
            deaths_df_pc.loc[ba_row, ba_col] = deaths[ba_row]["results_BA"][deaths[ba_row]["results_BA"]["BACODE"]==ba_col]["Deaths_pc"].values[0]
    
    #Make deaths per TWh imported/generated (in total) matrix
    deaths_df_pe = pd.DataFrame(index = importing_codes, columns = codes)
    for ba in importing_codes:
        #print(ba)
        TWh = deaths[ba]["generation"].sum()/1e6
        print(TWh)
        deaths_df_pe.loc[ba,:] = deaths_df.loc[ba,:]/TWh

  
    if csv_flag == True:
        deaths_df.to_csv(f'~/Desktop/IMSR/deaths{model_choice}_{CR}_df_coal2gas.csv')
        deaths_df_pc.to_csv('~/Desktop/IMSR/deathsAPSCA_K_df_pc_coal2gas.csv')
        
    matrices["total_deaths"] = deaths_df
    matrices["deaths_pc"] = deaths_df_pc
    matrices["deaths_pe"] = deaths_df_pe
    
    return matrices


# In[111]:


#BA_list = the list of responsible BAs to calculate results for
def calc_total_deaths_caused(BAs_sorted, results):
    BA_list = [BA for BA in BAs_sorted["BACODE"] if BA in results]
    #print(BA_list)
    total_deaths = {}
    deaths_twh = {}
    total_deaths_outside = {}
    deaths_twh_outside = {}
    
    for ba in BA_list:
        total_deaths[ba] = results[ba]["results_BA"]["Deaths"].sum()
        deaths_twh[ba] = results[ba]["results_BA"]["Deaths"].sum()/results[ba]["generation"].sum()*1e6
               
    #Calculate totals without self-damage
    total_deaths_outside = {}
    deaths_twh_outside = {}
    for ba in BA_list:
        total_deaths_outside[ba] = total_deaths[ba] - results[ba]["results_BA"][results[ba]["results_BA"]["BACODE"]==ba]["Deaths"].values[0]
        deaths_twh_outside[ba] = total_deaths_outside[ba]/results[ba]["generation"].sum()*1e6
    return total_deaths, deaths_twh, total_deaths_outside, deaths_twh_outside


# In[115]:


#codes: the list of balancing area codes to do the analysis for
def run_imports_and_self(BAs_sorted, model_choice,level, CR, coal_flag, gas_flag, coal2gas, CCS_coal, CCS_gas, CCS_coal2gas,results_imports, results_self):
    
    results_imports, errors_imports = count_deaths_all_bas(BAs_sorted, "import", model_choice, level, CR, coal_flag, gas_flag, coal2gas,CCS_coal, CCS_gas, CCS_coal2gas,results_imports)
    results_self, errors_self = count_deaths_all_bas(BAs_sorted, "self", model_choice, level, CR, coal_flag, gas_flag, coal2gas, CCS_coal, CCS_gas, CCS_coal2gas,results_self)
    
    
    return results_imports, results_self, errors_imports, errors_self


def process_imports_and_self(results_imports, results_self,model_choice,level,CR):
    BAs_sort = sort_BAs()
    #print('sorted BAs')
    #print(BAs_sort)
    
    matrices_imports = create_matrices(BAs_sort, results_imports, BAs_sort, list(results_imports.values())[0]["results_BA"]["BACODE"].drop_duplicates(), "ismr","L",False) 
    matrices_self = create_matrices(BAs_sort, results_self, BAs_sort, list(results_self.values())[0]["results_BA"]["BACODE"].drop_duplicates(), "ismr","L", False)
    #print('Created matrices')
    #print(matrices_imports["total_deaths"])
    
    total_deaths_imports, deaths_twh_imports, total_deaths_out_imports, deaths_twh_out_imports = calc_total_deaths_caused(BAs_sort, results_imports)
    total_deaths_self, deaths_twh_self, total_deaths_out_self, deaths_twh_out_self = calc_total_deaths_caused(BAs_sort, results_self)

    
    
    total_self = {}
    total_imports = {}
    total_self["total_deaths"] = total_deaths_self
    total_self["deaths_twh"] = deaths_twh_self
    total_self["total_out"] = total_deaths_out_self
    total_self["deaths_twh_out"] = deaths_twh_out_self
    total_imports["total_deaths"] = total_deaths_imports
    total_imports["deaths_twh"] = deaths_twh_imports
    total_imports["total_out"] = total_deaths_out_imports
    total_imports["deaths_twh_out"] = deaths_twh_out_imports
    
    return  matrices_imports, matrices_self, total_imports, total_self


# In[125]:


# #Run everything
# BAs_sorted = sort_BAs()
# #results_imports, results_self, errors_imports, errors_self = run_imports_and_self(BAs_sorted["BACODE"], "isrm","all","L", True, True, False, False, False, results_imports_coal, results_self_coal)
# results_imports_i, results_self_i, errors_imports_i, errors_self_i = run_imports_and_self(["CISO"], "isrm","all","LK", True, True, False, False, False, [], [])
# results_imports_i_K = {ba: results_imports_i[ba][0] for ba in results_imports_i}
# results_imports_i_L = {ba: results_imports_i[ba][1] for ba in results_imports_i}
# results_self_i_K = {ba: results_self_i[ba][0] for ba in results_self_i}
# results_self_i_L = {ba: results_self_i[ba][1] for ba in results_self_i}

# results_imports_a, results_self_a, errors_imports_a, errors_self_a = run_imports_and_self(["CISO"], "aspca_q0","all","LK", True, True, False, False, False, [], [])
# results_imports_a_K = {ba: results_imports_a[ba][0] for ba in results_imports_a}
# results_imports_a_L = {ba: results_imports_a[ba][1] for ba in results_imports_a}
# results_self_a_K = {ba: results_self_a[ba][0] for ba in results_self_a}
# results_self_a_L = {ba: results_self_a[ba][1] for ba in results_self_a}


# In[134]:


# results_imports_K = {ba: results_imports[ba][0] for ba in results_imports}
# results_imports_L = {ba: results_imports[ba][1] for ba in results_imports}
# results_self_K = {ba: results_self[ba][0] for ba in results_self}
# results_self_L = {ba: results_self[ba][1] for ba in results_self}


# In[137]:


#results_imports_K["CISO"]["results_county"]["Deaths"].sum()


# In[138]:


#results_imports_K["CISO"]["results_BA"]["Deaths"].sum()


# In[ ]:




