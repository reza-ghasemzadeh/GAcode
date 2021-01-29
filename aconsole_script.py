from simulation_model import CalibrationDataModel
from sys import argv


def set_params(individual, sim):
    param_dict = {'GKExperiment::reactionTimeAtt': 0,
                  'GKExperiment::reactionAtTrafficLightMesoAtt': 0,
                  'GKExperiment::capacityWeigthAtt': 0,
                  #'GKRoadType::distanceZone1Att': 0,
                  'GKTurning::lookaheadDistanceAtt': 0,
                  'GKRoadType::jamDensityAtt': 0}
    # assign generated values to dictionary
    for param, value in zip(param_dict, individual):
        param_dict[param] = value
    # assign experiment attributes
    exp_params = {attr: value for attr, value in param_dict.items() if attr.startswith('GKExperiment')}
    sim._setExperimentAttributes(exp_params)
    # assign road type attributes
    road_params = {attr: value for attr, value in param_dict.items() if attr.startswith('GKRoadType')}
    sim.setRoadAttributes(road_params)
    # assign turn specific attributes
    turn_params = {attr: value for attr, value in param_dict.items() if attr.startswith('GKTurning')}
    sim.setTurnAttributes(turn_params)
    
#     result_attr = sim._getAttributes(sim._getExperiment(), exp_params)
#     with open('GetExpParams.csv', 'w', newline='') as csvfile:
#         fieldnames = ['name', 'value']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for attr in result_attr:
#             writer.writerow(attr)
    return 0

if __name__ == '__main__':
    call_type = argv[1]
    if call_type == 'init':
        ang_file, rep_id, db_file = argv[2:]
        rep_id = int(rep_id)
        # initialize simulation model
        sim = CalibrationDataModel(angFile=ang_file, replicationId=rep_id)
        sim.loadNetwork()
        # specify database
        sim.setDatabase(db_file)
        # run the simulation for the first time
        sim.run()
        # unload the network
        sim.unloadNetwork()     #under line removed
    elif call_type == 'evaluate':
        ang_file, rep_id, db_file = argv[2:5]
        rep_id = int(rep_id)
        params = [float(arg) for arg in argv[5:]]
        # initialize simulation model
        sim = CalibrationDataModel(angFile=ang_file, replicationId=rep_id)   # new line added
        sim.loadNetwork()
        # specify database
        sim.setDatabase(db_file)
        # change the parameters
        set_params(params, sim)
        # run the simulation
        sim.run()
        # unload the network
        sim.unloadNetwork()   #under line removed