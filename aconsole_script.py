from simulation_model import CalibrationDataModel
from sys import argv


def set_params(individual, sim):
    param_dict = {'GKExperiment::reactionTimeAtt': 0,                     
                  'GKExperiment::reactionAtTrafficLightMesoAtt': 0,                 #new parameter
                  'GKExperiment::capacityWeigthAtt': 0,
                  'GKTurning::lookaheadDistanceAtt': 0,                             #new parameter
                  #'GKRoadType::lookaheadDistanceAtt': 0,
                  'GKSection::jamDensityAtt': 0}                                    #class change
                  #'GKRoadType::jamDensityAtt': 0}
    # assign generated values to dictionary
    for param, value in zip(param_dict, individual):
        param_dict[param] = value
    # assign experiment attributes
    exp_params = {attr: value for attr, value in param_dict.items() if attr.startswith('GKExperiment')}
    sim._setExperimentAttributes(exp_params)
    # assign road type attributes
    road_params = {attr: value for attr, value in param_dict.items() if attr.startswith('GKRoadType')}
    sim.setRoadAttributes(road_params)
    # assign turn attributes
    turn_params = {attr: value for attr, value in param_dict.items() if attr.startswith('GKTurning')}      #new line
    sim.setTurnAttributes(turn_params)       #new line
    # assign section attributes
    section_params = {attr: value for attr, value in param_dict.items() if attr.startswith('GKSection')}   #new line
    sim.setSectionAttributes(section_params) #new line
    #
    # Get Attributes
#     result_attr = sim._getAttributes(model.getCatalog().find(3193), param_dict)
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
        sim = CalibrationDataModel(angFile=ang_file, replicationId=rep_id)         #incorrenct angFile argument passed
        sim.loadNetwork()
        # specify database
        sim.setDatabase(db_file)                  # this function has self.loadNetwork() in it!
        # run the simulation for the first time
        sim.run()                                 # this function has self.loadNetwork() in it!
        # unload the network
        sim.unloadNetwork()                       # function calling corrected
    elif call_type == 'evaluate':
        ang_file, rep_id, db_file = argv[2:5]
        rep_id = int(rep_id)
        params = [float(arg) for arg in argv[5:]]
        # initialize simulation model
        sim = CalibrationDataModel(angFile=ang_file, replicationId=rep_id)   # new line added
        sim.loadNetwork()
        # specify database
        sim.setDatabase(db_file)                  # this function has self.loadNetwork() in it!
        # change the parameters
        set_params(params, sim)
        # run the simulation
        sim.run()                                 # this function has self.loadNetwork() in it!
        # unload the network
        sim.unloadNetwork()                       # function calling corrected