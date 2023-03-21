import json
import sys
import os
import math
from collections import OrderedDict
import pandas as pd
import datetime
from utility_functions import *
from jsonschema import Draft4Validator, validators, exceptions


class Profiler:
    def __init__(self, config):
        """
        :param baseline_config_files: list of strings to json files containing the base configuration regarding the function we want to run
        """
        self.optimization_iterations = config["optimization_iterations"]
        self.profiling_file = config["profiling_file"]
        if config["append_profiles"] and os.path.exists(self.profiling_file):
            self.append_profiles = True
        else:
            self.append_profiles = False
        self.name = config["application_name"]
        self.average_results = {self.name: {}}
        self.results = {}
        self.start_time = None

    def run(self):
        self.start_time = datetime.datetime.now()

    def stop(self):
        runtime = (datetime.datetime.now() - self.start_time).total_seconds()

        for key, value in self.results.items():
            self.average_results[self.name][key] = sum(value) / float(runtime)
            self.average_results[self.name]["Runtime per iteration (sec)"] = (
                runtime / self.optimization_iterations
            )

        if self.append_profiles:
            old_profile = pd.read_csv(self.profiling_file, index_col=0).to_dict()
            if self.has_same_keys(old_profile, self.average_results):
                for key, value in self.average_results.items():
                    identical_run_counter = 0
                    naming_key = key
                    while naming_key in old_profile.keys():
                        identical_run_counter += 1
                        naming_key = key + str(identical_run_counter)
                    old_profile[naming_key] = value

                profile_df = pd.DataFrame(old_profile)
                profile_df.to_csv(self.profiling_file)
            else:
                print(
                    "Could not extend current profile to old one due to header conflicts. Rename the output file or disable append_profiles"
                )
                print("The current profiling run will not have an output.")
                return

        else:
            profile_df = pd.DataFrame(self.average_results)
            profile_df.to_csv(self.profiling_file)

    def add(self, message, time):
        """
        adds a profiling checkpoint and the associated time, and stores it in self.results
        :param message: the name of the checkpoint in the .csv output file, along the lines of "Acqusition function eval. time"
        :param time: the time in seconds that the relevant secion has taken to complete
        """
        if message not in self.results.keys():
            self.results[message] = [time]

        self.results[message].append(time)

    # check that the old and new profile has the same measurement headers to enable putting them in the same file
    def has_same_keys(self, old_profile, new_profile):
        for result_old, result_new in zip(old_profile.values(), new_profile.values()):
            return set(result_old) == set(result_new)
