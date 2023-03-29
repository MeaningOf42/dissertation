from timeit import timeit
import platform
import cpuinfo
import audio_files
import audio_processing
import numpy as np
from logtools import printVerbose
import logtools
import time
import psutil
import git
import os

resultsFilename = "preformanceResults.csv"
logtools.verbose = "DETAIL"

# Preformaance Test Suite is a class that helps run a series of preformance related tests, identify the specifications of the
# system running the tests in order to eliminate error caused by running the code on faster or slower systems and saves all this
# information to a .csv file
class PreformanceTestSuite:
  
  # Save and Print system specs and the last git commit when you initialise a test suite
  def __init__(self):
    self.system = platform.system()
    self.systemVersion = platform.release()
    self.processor = '"' + platform.processor() + '"'
    self.cpuSpeed = cpuinfo.cpuinfo.get_cpu_info()['hz_actual'][0]
    print("Brand raw", cpuinfo.cpuinfo.get_cpu_info()["brand_raw"])
    self.cpuAvailible = 100-psutil.cpu_percent(4)
    self.memoryAvailible = psutil.virtual_memory()[1]/10**9

    repo = git.Repo(os.getcwd())
    master = repo.head.reference
    self.branch = master.name
    self.commitID = master.commit.hexsha
    self.commitMessage = master.commit.message

    self.tests = {}

    printVerbose(f"branch: {self.branch} commit: {self.commitID} commit message: {self.commitMessage}", detailLvl=4)
    printVerbose(f"OS: {self.system} version: {self.systemVersion}", detailLvl=4)
    printVerbose(f"Processor: {self.processor}  Speed: {self.cpuSpeed}", detailLvl=4)
    printVerbose(f"CPU availible: {self.cpuAvailible}gb", detailLvl=4)
    printVerbose(f"memory availible: {self.memoryAvailible}gb", detailLvl=4)

  # Runs a function multiple times and save how long it take to run on average,
  # and how many times it ran to the self.tests dictionary.
  # The function will be ran reps times, unless after one iteration it is estimated that running it this many times will take
  # longer than maxTime (in seconds). In this case it will test the function the maximum number of times it estimates it can run
  # with maxTime.
  def runTest(self, testName, func, reps = 1000, maxTime = 10):
    singleTime = timeit(func, number = 1)
    if singleTime*reps > maxTime:
      newReps = int(maxTime//singleTime)
      printVerbose(testName, f"would take significantly longer than max time: {maxTime}, to run {reps}",
                   "times, instead limiting the test to only run", newReps, "times", detailLvl="DETAIL")
      reps = newReps

    printVerbose(f"running '{testName}' {reps} times, ", end="")

    repsTime = timeit(func, number=reps-1)+singleTime
    averageTime = repsTime / reps
    self.tests[testName] = (averageTime, reps)
    printVerbose(f"average time: {averageTime:.6f}s")

  # Stores the results of all the tests ran by the test suite along with system information in a .csv
  def logResults(self):
    with open(resultsFilename, "a") as resultsFile:
      for testName, (averageTime, reps) in self.tests.items():
        print(int(time.time()), self.branch, self.commitID, self.commitMessage,
              self.system, self.systemVersion, self.processor,
              self.cpuSpeed, self.cpuAvailible, self.memoryAvailible,
              testName, reps, averageTime, sep=",", file=resultsFile)
    
    
# initialises a premformance test suite object
suite = PreformanceTestSuite()

# run a series of preformance tests that involve opening an audio file, reading it's spectrogram,
# and evaluating an audio processing chain.

suite.runTest("open test file", lambda: audio_files.read_wave_file("stereo_test.wav"))

stereoTest = audio_files.read_wave_file("stereo_test.wav")
suite.runTest("create Spectrogram", lambda: stereoTest.calculate_spectrogram(0.1))

sinPowGen = audio_processing.WaveMixGen("main")
suite.runTest("evaluate SinPowGen with random params",
              lambda: sinPowGen.arrayToAudio(44000, 2, np.random.random(6)))

sinGen = audio_processing.SinGen("main")
sinGen.setArgument("amp", audio_processing.SinGen("vol mod"))
freqControl = audio_processing.AddNode("freq mod")
freqControl.setArgument("a", audio_processing.SinGen("freq osc"))
sinGen.setArgument("freq", freqControl)

printVerbose()
printVerbose("complex audio processing chain: ")
printVerbose(sinGen.ArrayLabelsStr())

printVerbose()
suite.runTest("evaluate complex audio processing chain w random params",
              lambda: sinGen.arrayToAudio(44000, 2, np.random.random(8)*100))

# stores the results of the tests to resultsFilename (a .csv file)
suite.logResults()
