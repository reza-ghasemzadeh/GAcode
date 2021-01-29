import logging
import os
import time
from PyANGBasic import *
from PyANGKernel import *
from PyANGConsole import *
from PyQt5 import *


class CalibrationDataModel(object):
	""" A class for a microscopic Aimsun model to generate data required for 
		deep-learning-based calibration"""
	
	def __init__(self, angFile, replicationId, console=None, model=None):
		self._angFile = angFile
		self._id = replicationId
		self._model = model
		self._console = console
		self._logger = logging.getLogger(__name__)

	def _checkValidId(self):
		return self._id > 0

	def loadNetwork(self):
		# start console
		self._logger.debug('Loading %s' %self._angFile)
		if self._console is not None and self._model is not None:
			self._logger.debug('AIMSUN is already loaded')
		else:
			if self._console is None:
				self._console = ANGConsole()
			# load a network
			if self._console.open(self._angFile):
				self._logger.debug('Loading Aimsun Model')
				self._model = self._console.getModel()
			else:
				self._logger.error('cannot load network')
				self._console = None
				self._model = None

		return (self._console, self._model)   #why it has paranthesis?

	def unloadNetwork(self):
		# close Aimsun
		if self._console is not None:
			try:
				self._logger.debug('Closing Aimsun')
				self._console.close()
				self._model = self._console.getModel()     #why getting the model after closing?
			except:
				print('ERROR: cannot close AIMSUN')
		else:
			self._logger.error('No Aimsun instance is running')
		return (self._console, self._model)     #why it has paranthesis?

	def _getExperiment(self):
		replication = self._model.getCatalog().find(self._id)
		return replication.getExperiment()    # is it right?

	def _getScenario(self):
		return self._getExperiment().getScenario()  #is it right? it should not be "self.getScenario().getExperiment()"?

	def setDatabase(self, databaseName=None):
		if self._console is None or self._model is None:
			self.loadNetwork()
		if databaseName is not None:
			self._database = databaseName
		if self._database == '':
			return
		scenario = self._getScenario()
		# get the scenario's DB info
		dbInfo = scenario.getDB(False)
		# custom DB
		dbInfo.setUseProjectDB(False)
		dbInfo.setAutomatic(False)
		# create if it doesn't exist
		dbInfo.setAutomaticallyCreated(True)
		# SQLite DB
		dbInfo.setDriverName('QSQLITE')
		# DB Full Name
		dbInfo.setDatabaseName(self._database)
		# assign it to the scenario
		scenario.setDB(dbInfo)
		self._logger.debug('Database of scenario %i has been changed to %s' % (scenario.getId(), self._database))

	def _getDatabase(self):
		return self._database

	def removeDatabase(self):
		try:
			os.remove(self._database)
			self._logger.debug('%s has been removed' % self._database)
		except OSError:
			self._logger.debug('Cannot remove database: %s' % self._database)

	def _setAttributes(self, object, attributes):  #the "object" won't be confused with the class object imputed at beginning?
		# attributes --> {att : val}
		objType = object.getType()
		for key in attributes: #OR: for key, value in attributes.items(): object.setDataValue(key, value)
			att = objType.getColumn(key, GKType.eSearchOnlyThisType)
			object.setDataValue(att, attributes[key])

	def _setExperimentAttributes(self, attributes):
		self._setAttributes(self._getExperiment(), attributes)

	def setRoadAttributes(self, attributes):
		# road's parameters
		roadType = self._model.getType("GKRoadType")
		roadTypes = self._model.getCatalog().getObjectsByType(roadType) #space in the empty paranthesis removed
		if roadTypes != None:
			for roadType in roadTypes.values():
				self._setAttributes(roadType, attributes) #it gets the type of the type of an object!!!

	def setTurnAttributes(self, attributes):
		# turn's parameters
		turnType = self._model.getType("GKTurning")
		turnTypes = self._model.getCatalog().getObjectsByType(turnType) #space in the empty paranthesis removed
		if turnTypes != None:
			for turnType in turnTypes.values():
				self._setAttributes(turnType, attributes) #it gets the type of the type of an object!!!

	def _getAttributes(self, object, attributes):
		# attributes --> [att]
		# result --> {att : val}
		result = {}
		objType = object.getType()
		for key in attributes:
			att = objType.getColumn(key, GKType.eSearchOnlyThisType)
			if object.getDataValue(att)[1]:
				result[key] = object.getDataValue(att)[0]  # index [0] is right? it should not be 1?
		return result

	def run(self):
		'''
		load the model if it's not already loaded
		and run the model
		'''
		self.loadNetwork()
		if self._model is not None:
			replication = self._model.getCatalog().find(self._id)
			if replication is None or not replication.isA("GKReplication"):
				self._console.getLog().addError("Cannot find replication")
				self._logger.error("Cannot find replication %i" % self._id)
				return -1
			else:
				self.removeDatabase()
				experiment = self._getExperiment()
				scenario = self._getScenario()
				# run the simulation
				selection = []
				self._logger.debug('Starting simulation %i' % self._id)
				GKSystem.getSystem().executeAction("execute", replication,
										selection, time.strftime("%Y%m%d-%H%M%S"))
				return 0