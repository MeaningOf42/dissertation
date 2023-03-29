import datetime

verbose = "VERBOSE"
outputFile = "log.txt"
logging = False

detailLevels = {
  "SILENT": 0,
  "ERROR": 1,
  "WARNING": 2,
  "VERBOSE": 3,
  "DETAIL": 4,
  "DEBUG": 5
  }


def printVerbose(*args, detailLvl="VERBOSE", **kwargs):
  global verbose
  if type(verbose) == str:
    verbose = detailLevels[verbose]
  if type(detailLvl) == str:
    detailLvl = detailLevels[detailLvl]
  if verbose >= detailLvl:
    print(*args, **kwargs)
  if logging:
    with open(outputFile, 'a') as f:
      print(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            f"[{list(detailLevels.keys())[detailLvl]}]:  ",
            *args, file=f, **kwargs)
