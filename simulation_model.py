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
		return (self._console, self._model)

	def unloadNetwork(self):
		# close Aimsun
		if self._console is not None:
			try:
				self._logger.debug('Closing Aimsun')
				self._console.save()
				self._console.close()
				self._model = None      #self._console.getModel()
			except:
				print('ERROR: cannot close AIMSUN')
		else:
			self._logger.error('No Aimsun instance is running')
		return (self._console, self._model)

	def _getExperiment(self):
		replication = self._model.getCatalog().find(self._id)
		return replication.getExperiment()

	def _getScenario(self):
		return self._getExperiment().getScenario()

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

	def _setAttributes(self, obj, attributes):   #1st arg name changed
		# attributes --> {att : val}
		objType = obj.getType()
		for key in attributes:
			att = objType.getColumn(key, GKType.eSearchOnlyThisType)
			obj.setDataValue(att, attributes[key])

	def _setExperimentAttributes(self, attributes):
		self._setAttributes(self._getExperiment(), attributes)

	def setRoadAttributes(self, attributes):
		# road's parameters
		roadType = self._model.getType("GKRoadType")
		roadTypes = self._model.getCatalog( ).getObjectsByType(roadType)
		if roadTypes != None:
			for roadType in roadTypes.values():
				self._setAttributes(roadType, attributes)

	def setTurnAttributes(self, attributes):         #New Function
		# turn's parameters
		turnType = self._model.getType("GKTurning")
		turnTypes = self._model.getCatalog( ).getObjectsByType(turnType)
		if turnTypes != None:
			for turnType in turnTypes.values():
				GKTurning().setUseRoadTypeDistanceZones(False)  #New line
				self._setAttributes(turnType, attributes)

	def setSectionAttributes(self, attributes):      #New Function
		# section's parameters
		sectionType = self._model.getType("GKSection")
		sectionTypes = self._model.getCatalog( ).getObjectsByType(sectionType)
		if sectionTypes != None:
			for sectionType in sectionTypes.values():
				self._setAttributes(sectionType, attributes)

	def _getAttributes(self, obj, attributes):     #1st arg name changed
		# attributes --> [att]
		# result --> {att : val}
		result = {}
		objType = obj.getType()
		for key in attributes:
			att = objType.getColumn(key, GKType.eSearchOnlyThisType)
			if obj.getDataValue(att)[1]:
				result[key] = obj.getDataValue(att)[0]
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
				#self.removeDatabase()
				experiment = self._getExperiment()   #why we need this varialble?
				scenario = self._getScenario()       #why we need this varialble?
				# run the simulation
				selection = []
				self._logger.debug('Starting simulation %i' % self._id)
				GKSystem.getSystem().executeAction("execute", replication,
										selection, time.strftime("%Y%m%d-%H%M%S"))
				return 0