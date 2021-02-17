'''
# init scoop
import runpy
runpy.run_module('scoop')
'''
# Utilities
import random
import os
import shutil
import tempfile
import sys
import site
import logging
import time
import subprocess
import csv

#sys.path.append('C:\Program Files\Python38\Lib\site-packages')
sys.path.append('C:\\Program Files\\Aimsun\\Aimsun Next 20\\Lib\\site-packages')
sys.path.append('C:\\Program Files\\Aimsun\\Aimsun Next 20')

# configure logger
# define log filename
LOG_FILENAME = 'C:\\Aimsun Projects\\REZA_GA\\calibration.out'                             #Directory changed
logger = logging.getLogger("my logger")
logger.setLevel(logging.DEBUG)
# format for our loglines
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# Setup file logging
fh = logging.FileHandler(LOG_FILENAME)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# Genetic Algorithm
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import math

# parallelization
import scoop
from scoop import futures
import multiprocessing as mp

# Data Processing
import numpy as np
import sqlite3
import pandas as pd
# silence pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

# define fitness weights
weights = (-0.5, -0.5)
# create fitness min class - inherits deap's base fitness class, uses defined weights
creator.create('FitnessMin', base.Fitness, weights=weights)
# create individual class - inherits list, uses FitnessMax class in fitness attr
creator.create('Individual', list, fitness=creator.FitnessMin)

# init toolbox
toolbox = base.Toolbox()

def run_aimsun_command(args):
    '''
    Script is one of 2 types:
    1. init - intialize an aimsun network, specifying the angfile, rep_id and db file
    2. set params and run - run an aimsun network, specifying same as init, but additionally with parameters
    '''
    # build the command
    cmd = [aimsun_exe, '--script']      #a hyphen  added before '-script'
    cmd.append(os.path.normpath(os.path.join(os.path.dirname(__file__),'aconsole_script.py')))
    cmd += args
    print(cmd)            #new line added
    # execute the command
    ps = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # wait for command to finish
    ps.wait()
    return 0

def to_dict(db_file):
    '''
    converts the tables within the supplied sqlite file into a dictionary
    with table names as keys and dataframe objects as values
    '''
    db = sqlite3.connect(db_file)
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_dict = {}
    for table_name in tables:
        table_name = table_name[0]
        table = pd.read_sql_query("SELECT * from %s" % table_name, db)
        table_dict[table_name] = table
    cursor.close()
    db.close()
    return table_dict

def reindex_obs_data(obs_data, sim_info):
    '''
    reindexes observed data to align with simulated data
    '''
    # obtain simulation time information
    start_time_s = sim_info['from_time'][0] # (time in seconds)
    start_time_h = start_time_s//3600 # start time hour
    start_time_m = (start_time_s % 3600)//60 # start time minute
    duration = tables['SIM_INFO']['duration'][0] # duration in seconds
    num_intervals = sim_data['ent'].nunique() - 1
    interval_len = duration/num_intervals # length in seconds
    end_time_s = start_time_s + duration
    end_time_h = end_time_s//3600 # start time hour
    end_time_m = (end_time_s % 3600)//60 # start time minute

    # get only observed data within simulation time window
    time_mask = (((pd.to_datetime(obs_data['time']).dt.hour == start_time_h) &
              (pd.to_datetime(obs_data['time']).dt.minute >= start_time_m)) |
              ((pd.to_datetime(obs_data['time']).dt.hour > start_time_h) &
                (pd.to_datetime(obs_data['time']).dt.hour < end_time_h)) |
                ((pd.to_datetime(obs_data['time']).dt.hour == end_time_h) &
                (pd.to_datetime(obs_data['time']).dt.minute <= end_time_m)))
    obs_data_reindexed = obs_data[time_mask]
    # re-index intervals to match simulation data, observed data is on 15 minute itervals, only need 30 minute intervals to match simulated data
    obs_data_reindexed = obs_data_reindexed.reset_index().drop('index',axis=1)
    # only take rows that are within 30 minute intervals
    obs_data_reindexed = obs_data[(obs_data['time'].apply(lambda x: x.split(':')[1]) == '30') | (obs_data['time'].apply(lambda x: x.split(':')[1]) == '00')]
    # create interval column to join both datasets
    obs_data_reindexed['ent'] = ''
    for i, time in enumerate(obs_data_reindexed['time'].unique()):
        obs_data_reindexed.loc[obs_data.time==time, 'ent'] = i
    obs_data_reindexed.to_csv('obs_data_reindexed.csv')                #new line
    return obs_data_reindexed

def get_sim_data(tables):
    # get base detector data
    sim_data = tables['MEDETECT']
    # take aggregated vehicle type data, and non-aggregated interval
    sim_data = sim_data[(sim_data['sid']==0)&(sim_data['ent']!=0)]
    return sim_data

def join_obs_sim(obs_data, sim_data):
    # only take relevant column from each dataset
    req_columns = ['eid','ent', 'flow','speed']
    obs_data = obs_data[req_columns]
    sim_data = sim_data[req_columns]
    # rename columns in datasets for identification once joined
    obs_data.rename(columns={'flow': 'flow_obs', 'speed': 'speed_obs'}, inplace=True)
    sim_data.rename(columns={'flow': 'flow_sim', 'speed': 'speed_sim'}, inplace=True)
    # join tables
    joined_data = obs_data.merge(sim_data, how='inner', on=['eid', 'ent'])

    return joined_data

aimsun_exe = 'C:\\Program Files\\Aimsun\\Aimsun Next 20\\aconsole.exe'
# define paths
data_path = 'C:\\Users\\RezaGS\\Desktop\\Final Edition\\Testing\\Testing\\Outputs Tutorial\\Model\\Resources\\Outputs\\'
db_file = data_path + 'results_aimsun.sqlite'
obs_data_file = data_path + 'ObsDataSynthesized.csv'  #obs file replaced
aimsun_file = 'C:\\Users\\RezaGS\\Desktop\\Final Edition\\Testing\\Testing\\Outputs Tutorial\\Model\\Final_Outputs_new.ang'
rep_id = 3194

# initialize simulation model
run_aimsun_command(['init', aimsun_file, str(rep_id), db_file])

# create dictionary of db tables
tables = to_dict(db_file)

# get simulation info
sim_info = tables['SIM_INFO']

# get observed data
obs_data = pd.read_csv(obs_data_file)  #double 'obs_data =' removed

# get simulated data
sim_data = get_sim_data(tables)

# re index obeserved data to match indexing of simulated data
obs_data = reindex_obs_data(obs_data, sim_info)
# remove observed data with zero flows
obs_data = obs_data[obs_data['flow'] > 0]

# join sim and obs data
joined_data = join_obs_sim(obs_data, sim_data)
# establish actual flows and speeds for error calculations
flows_actual = joined_data['flow_obs']
speeds_actual = joined_data['speed_obs']

#in AIMSUN help the formula is different: the RMSE is based on the percentage error: RMSE=sqrt( 1/N * sum[((sim-obs)/obs)**2] )
def mean_squared_error(y_true, y_pred, squared=True):
    MSE = np.square(np.subtract(y_true,y_pred)).mean()
    if squared:
        return MSE
    else:
        return math.sqrt(MSE)

def to_csv(db_file):
    '''
    converts the tables within the supplied sqlite file into csvs
    '''
    db = sqlite3.connect(db_file)
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table_name in tables:
        table_name = table_name[0]
        table = pd.read_sql_query("SELECT * from %s" % table_name, db)
        export_path = db_file.rsplit('/', 1)[0]
        table.to_csv(export_path + '/' + table_name + '.csv', index_label='index')
    cursor.close()
    db.close()

# Evaluation function
def mean_geh(y_true, y_pred):
    geh_vector = np.sqrt( (2*(y_pred - y_true)**2) / (y_true + y_pred) )
    return np.mean(geh_vector)

def simulate(individual, ang_file, db_file):
    # convert individual to string so it can be passed as a command line argument
    string_individual = [str(x) for x in individual]
    run_aimsun_command(['evaluate', ang_file, str(rep_id), db_file] + string_individual)
    # create dictionary of db tables
    tables = to_dict(db_file)
    # get new simulated results
    sim_data = get_sim_data(tables)

    # join sim and obs data
    joined_data = join_obs_sim(obs_data, sim_data)

    # get simulation flow and speed values
    flows_simulated = joined_data['flow_sim']
    speeds_simulated = joined_data['speed_sim']

    return flows_simulated, speeds_simulated

def eval(individual):
    #print('Evaluating individual...')
    # create a temporary copy of the ang file to evaluate in parallel
    temp_ang, temp_db = make_temp_sim()
    # run the simulation
    flows_simulated, speeds_simulated = simulate(individual, temp_ang, temp_db)
    # remove temporary files
    #os.remove(temp_ang)
    #os.remove(temp_db)
    # compute error metrics
    geh = mean_geh(flows_actual, flows_simulated)
    rmse = mean_squared_error(speeds_actual, speeds_simulated, squared=False)
    #print('Evaluation complete!')
    return rmse, geh

# register evaluation function
toolbox.register("evaluate", eval)
# register mating function
toolbox.register("mate", tools.cxTwoPoint)
# register mutation function
#toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
low = [0.8, 1.4, 0.01, 280, 100]  #lower bound for each gene
up = [2, 2, 5, 420, 200]    #upper bound for each gene
toolbox.register('mutate', tools.mutPolynomialBounded, eta=0.5, low=low, up=up, indpb=0.1)
# register selection function
toolbox.register("select", tools.selTournament, tournsize=3)

# get number of threads/workers
num_workers = 10 #mp.cpu_count()     #update
toolbox.register("map", futures.map)

def create_temporary_copy(src_file_name, preserve_extension=False):
    '''
    Copies the source file into a temporary file.
    Returns a _TemporaryFileWrapper, whose destructor deletes the temp file
    (i.e. the temp file is deleted when the object goes out of scope).
    '''
    tf_suffix=''
    dir_name, _ = os.path.split(src_file_name)
    if preserve_extension:
        _, tf_suffix = os.path.splitext(src_file_name)
    # generate a temporary file name
    tf = tempfile.NamedTemporaryFile(suffix=tf_suffix,dir=dir_name, delete=False)
    # create a temporary copy of the source file
    shutil.copy(src_file_name, tf.name)
    return tf.name

def make_temp_sim():
    # create temp files
    #print('Creating temporary simulation...')
    temp_ang = create_temporary_copy(aimsun_file, preserve_extension=True)
    temp_db = create_temporary_copy(db_file, preserve_extension=True)

    return temp_ang, temp_db

# def uniform():                              #unnecessary function removed!
#     return random.uniform(0, 20)

def random_generator(start, end, decimal):     #new fuction
    decim=10**decimal
    random_number=random.randint(start*decim,end*decim)/decim
    return random_number

def get_individuals(indlist):                  #new fuction
    genes_values=[]
    for ind in indlist:
        genes_values.append(ind)
    return genes_values

def get_finesses(indlist):                     #new fuction
    fitness_values=[]
    for ind in indlist:
        fitness_values.append(ind.fitness.values)
    return fitness_values

def get_finesses_mean(indlist):                #new fuction
    fitness_values=[]
    for ind in indlist:
        fitness_values.append(ind.fitness.values)
    mean=np.mean(fitness_values, axis=0)
    return mean

def get_finesses_std(indlist):                 #new fuction
    fitness_values=[]
    for ind in indlist:
        fitness_values.append(ind.fitness.values)
    std=np.std(fitness_values, axis=0)
    return std

def get_finesses_min(indlist):                 #new fuction
    fitness_values=[]
    for ind in indlist:
        fitness_values.append(ind.fitness.values)
    mins=np.min(fitness_values, axis=0)
    return mins

def get_finesses_max(indlist):                 #new fuction
    fitness_values=[]
    for ind in indlist:
        fitness_values.append(ind.fitness.values)
    maxs=np.max(fitness_values, axis=0)
    return maxs

def main():
    # define cross and mutation probability constants
    CXPB, MUTPB = 0.7, 0.1      #update

    # attribute generator - random float generator (start, end, decimal)                       #update
    toolbox.register('attr_param1', random_generator, 0.8, 2, 2)    #reaction time
    toolbox.register('attr_param2', random_generator, 1.4, 2, 2)    #reation at traffic light
    toolbox.register('attr_param3', random_generator, 0.01, 5, 2)   #capacityWeight
    toolbox.register('attr_param4', random_generator, 280, 420, 2)  #Look-Ahead Distance
    toolbox.register('attr_param5', random_generator, 100, 200, 2)  #Jam Density
    #
    # register individual function - 5 parameters
    toolbox.register('individual', tools.initCycle, creator.Individual, (toolbox.attr_param1, toolbox.attr_param2, toolbox.attr_param3, toolbox.attr_param4, toolbox.attr_param5), n=1)                  #update
    #
    # register population function - list of previously registered individuals
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # init the population, size being equal to num of cpu cores ' num_workers '
    pop = toolbox.population(n=10)                                                    #update
    #
    # init hall of fame object, to hold the 10 best individuals
    hof = tools.HallOfFame(10)                                                        #update
    #
    # init stats object
    #stats = tools.Statistics(lambda ind: ind.fitness.values)
    #stats.register("avg", np.mean, axis=0)
    #stats.register("std", np.std, axis=0)
    #stats.register("min", np.min, axis=0)
    #stats.register("max", np.max, axis=0)
    #
    # init stats object                                                               #update
    stats = tools.Statistics()
    stats.register("avg", get_finesses_mean)
    stats.register("std", get_finesses_std)
    stats.register("min", get_finesses_min)
    stats.register("max", get_finesses_max) 
    stats.register("fitnesses", get_finesses)
    stats.register("individuals", get_individuals)
    #
    # Algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=10,
                                   stats=stats, halloffame=hof, verbose=True)
    return pop, log, hof


if __name__ == '__main__':
    pop, log, hof = main()
    print('----Individual with highest fitness----')
    param_dict = {'GKExperiment::reactionTimeAtt': 0,                     
                  'GKExperiment::reactionAtTrafficLightMesoAtt': 0,                 #new parameter
                  'GKExperiment::capacityWeigthAtt': 0,
                  'GKTurning::lookaheadDistanceAtt': 0,                             #new parameter
                  #'GKRoadType::lookaheadDistanceAtt': 0,
                  'GKSection::jamDensityAtt': 0}                                    #class change
                  #'GKRoadType::jamDensityAtt': 0}
    # assign generated values to dictionary
    for param, value in zip(param_dict, hof[0]):
        print('{}: {}'.format(param, value))
    #    
    # Hall of Fame 
    print('----Best individuals over all generations----')
    bests =[['gene1', 'gene2', 'gene3', 'gene4', 'gene5', 'fit1', 'fit2']]
    for ind in hof:
        print('Individual:{}, Fitness Values:{}'.format(ind, ind.fitness.values))
        ind.append(ind.fitness.values[0])
        ind.append(ind.fitness.values[1])
        bests.append(ind)        
    #
    # gives the Hall of Fame in csv format
    with open("HallOfFame.csv", "w", newline="") as fame:
        writer = csv.writer(fame)
        writer.writerows(bests)
    #
    # gives the LogBook in csv format
    with open('LogBook.csv', 'w', newline='') as csvfile:
        fieldnames = ['gen', 'nevals', 'avg', 'std', 'min', 'max', 'fitnesses', 'individuals']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for gen in log:
            writer.writerow(gen)
    #
    # History
    hist=[['gene1', 'gene2', 'gene3', 'gene4', 'gene5', 'RMSE', 'GEH', 'Weightened Fits']]
    for gen in log:
        for chromo in gen['individuals']:
            #print('Individual:{}, Fitness Values:{}'.format(chromo, chromo.fitness.values))
            chromo.append(chromo.fitness.values[0])
            chromo.append(chromo.fitness.values[1])
            chromo.append(np.average(chromo.fitness.values, weights=weights))
            hist.append(chromo)
        emp=['', '', '', '', '', '', '', '']
        hist.append(emp)
    # gives the History of GA in csv format
    with open("History.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(hist)
    #       
    # Final Population
    print('----Final Population and corresponding fitness values----')
    finalpop=[['gene1', 'gene2', 'gene3', 'gene4', 'gene5', 'RMSE', 'GEH', 'Weightened Fits']]
    for indiv in pop:
        print('Individual:{}, Fitness Values:{}'.format(indiv[:5], indiv[5:]))
         #indiv.append(indiv.fitness.values[0])
         #indiv.append(ind.fitness.values[1])
        finalpop.append(indiv)
    # gives the Final Population in csv format
    with open("FinalPop.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(finalpop)


