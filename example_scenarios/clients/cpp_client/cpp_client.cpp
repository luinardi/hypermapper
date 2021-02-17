#include <cstdint>
#include <experimental/filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "cpp_client.h"
#include "json.hpp"

using namespace std;

using json = nlohmann::json;
namespace fs = experimental::filesystem;

// popen2 implementation adapted from:
// https://github.com/vi/syscall_limiter/blob/master/writelimiter/popen2.c
struct popen2 {
  pid_t child_pid;
  int from_child, to_child;
};

int popen2(const char *cmdline, struct popen2 *childinfo) {
  pid_t p;
  int pipe_stdin[2], pipe_stdout[2];

  if (pipe(pipe_stdin))
    return -1;
  if (pipe(pipe_stdout))
    return -1;

  printf("pipe_stdin[0] = %d, pipe_stdin[1] = %d\n", pipe_stdin[0],
         pipe_stdin[1]);
  printf("pipe_stdout[0] = %d, pipe_stdout[1] = %d\n", pipe_stdout[0],
         pipe_stdout[1]);

  p = fork();
  if (p < 0)
    return p;   /* Fork failed */
  if (p == 0) { /* child */
    close(pipe_stdin[1]);
    dup2(pipe_stdin[0], 0);
    close(pipe_stdout[0]);
    dup2(pipe_stdout[1], 1);
    execl("/bin/sh", "sh", "-c", cmdline, 0);
    perror("execl");
    exit(99);
  }
  childinfo->child_pid = p;
  childinfo->to_child = pipe_stdin[1];
  childinfo->from_child = pipe_stdout[0];
  return 0;
}

void fatalError(const string &msg) {
  cerr << "FATAL: " << msg << endl;
  exit(EXIT_FAILURE);
}

// Function that creates the json scenario for hypermapper
// Arguments:
// - AppName: Name of application
// - OutputFolderName: Name of output folder
// - NumIterations: Number of HP iterations
// - NumDSERandomSamples: Number of HP random samples
// - Predictor: Boolean for enabling/disabling feasibility predictor
// - InParams: vector of input parameters
// - Objectives: string with objective names
string createjson(string AppName, string OutputFoldername, int NumIterations,
                  int NumDSERandomSamples, bool Predictor,
                  vector<HMInputParamBase *> &InParams, vector<string> Objectives) {

  string CurrentDir = fs::current_path();
  string OutputDir = CurrentDir + "/" + OutputFoldername + "/";
  if (fs::exists(OutputDir)) {
    cerr << "Output directory exists, continuing!" << endl;
  } else {

    cerr << "Output directory does not exist, creating!" << endl;
    if (!fs::create_directory(OutputDir)) {
      fatalError("Unable to create Directory: " + OutputDir);
    }
  }
  json HMScenario;
  HMScenario["application_name"] = AppName;
  HMScenario["optimization_objectives"] = json(Objectives);
  HMScenario["hypermapper_mode"]["mode"] = "client-server";
  HMScenario["run_directory"] = CurrentDir;
  HMScenario["log_file"] = OutputFoldername + "/log_" + AppName + ".log";
  HMScenario["optimization_iterations"] = NumIterations;
  HMScenario["models"]["model"] = "random_forest";

  if (Predictor) {
    json HMFeasibleOutput;
    HMFeasibleOutput["enable_feasible_predictor"] = true;
    HMFeasibleOutput["false_value"] = "0";
    HMFeasibleOutput["true_value"] = "1";
    HMScenario["feasible_output"] = HMFeasibleOutput;
  }

  HMScenario["output_data_file"] =
      OutputFoldername + "/" + AppName + "_output_data.csv";
  HMScenario["output_pareto_file"] =
      OutputFoldername + "/" + AppName + "_output_pareto.csv";
  HMScenario["output_image"]["output_image_pdf_file"] =
      OutputFoldername + "_" + AppName + "_output_image.pdf";

  json HMDOE;
  HMDOE["doe_type"] = "standard latin hypercube"; // "random sampling";
  HMDOE["number_of_samples"] = NumDSERandomSamples;

  HMScenario["design_of_experiment"] = HMDOE;

  for (auto InParam : InParams) {
    json HMParam;
    HMParam["parameter_type"] = getTypeAsString(InParam->getType());
    switch (InParam->getDType()) {
      case Int:
        HMParam["values"] = json(static_cast<HMInputParam<int>*>(InParam)->getRange()); 
        break;
      case Float:
        HMParam["values"] = json(static_cast<HMInputParam<float>*>(InParam)->getRange()); 
        break;
    }
    HMScenario["input_parameters"][InParam->getKey()] = HMParam;
  }

  //  cout << setw(4) << HMScenario << endl;
  ofstream HyperMapperScenarioFile;

  string JSonFileNameStr =
      CurrentDir + "/" + OutputFoldername + "/" + AppName + "_scenario.json";

  HyperMapperScenarioFile.open(JSonFileNameStr);
  if (HyperMapperScenarioFile.fail()) {
    fatalError("Unable to open file: " + JSonFileNameStr);
  }
  cout << "Writing JSON file to: " << JSonFileNameStr << endl;
  HyperMapperScenarioFile << setw(4) << HMScenario << endl;
  return JSonFileNameStr;
}

// Function that takes input parameters and generates objective
HMObjective calculateObjective(vector<HMInputParamBase *> &InputParams) {

  HMObjective Obj;
  float x1 = static_cast<HMInputParam<float>*>(InputParams[0])->getVal();
  int x2 = static_cast<HMInputParam<int>*>(InputParams[1])->getVal();

  Obj.f1_value = 2 + (x1 - 2) * (x1 - 2) + (x2 - 1) * (x2 - 1);
  Obj.f2_value = 9 * x1 - (x2 - 1) * (x2 - 1);

  bool c1 = ((x1 * x1 + x2 * x2) <= 255);
  bool c2 = ((x1 - 3 * x2 + 10) <= 0);
  Obj.valid = c1 && c2;
  return Obj;
}

// Function that populates input parameters
int collectInputParams(vector<HMInputParamBase *> &InParams) {
  int numParams = 0;

  vector<float> floatRange = {-20.0, 20.0};
  vector<int> intRange = {-20, 20};

  HMInputParam<float> *x0Param = new HMInputParam<float>("x0", ParamType::Real);
  x0Param->setRange(floatRange);
  InParams.push_back(x0Param);
  numParams++;

  HMInputParam<int> *x1Param = new HMInputParam<int>("x1", ParamType::Integer);
  x1Param->setRange(intRange);
  InParams.push_back(x1Param);
  numParams++;
  return numParams;
}

// Free memory of input parameters
void deleteInputParams(vector<HMInputParamBase *> &InParams) {
  for (auto p : InParams) {
    switch(p->getDType()) {
      case Int:
        delete static_cast<HMInputParam<int>*>(p);
        break;
      case Float:
        delete static_cast<HMInputParam<float>*>(p);
        break;
      default:
        fatalError("Trying to free unhandled data type.");
    }
  }
}


// Function for mapping input parameter based on key
auto findHMParamByKey(vector<HMInputParamBase *> &InParams, string Key) {
  for (auto it = InParams.begin(); it != InParams.end(); ++it) {
    HMInputParamBase Param = **it;
    if (Param == Key) {
      return it;
    }
  }
  return InParams.end();
}

//Function that sets the input parameter value
void setInputValue(HMInputParamBase *Param, string ParamVal) {
  switch(Param->getDType()) {
    case Int:
      static_cast<HMInputParam<int>*>(Param)->setVal(stoi(ParamVal));
      break;
    case Float:
      static_cast<HMInputParam<float>*>(Param)->setVal(stof(ParamVal));
      break;
  }
}

int main(int argc, char **argv) {

  if (!getenv("HYPERMAPPER_HOME")) {
    string ErrMsg = "Environment variables are not set!\n";
    ErrMsg += "Please set HYPERMAPPER_HOME before running this ";
    fatalError(ErrMsg);
  }

  srand(0);

  // Set these values accordingly
  // TODO: make these command line inputs
  string OutputFoldername = "outdata";
  string AppName = "cpp_chakong_haimes";
  int NumIterations = 20;
  int NumSamples = 10;
  bool Predictor = 1;
  vector<string> Objectives = {"f1_value", "f2_value"};

  // Create output directory if it doesn't exist
  string CurrentDir = fs::current_path();
  string OutputDir = CurrentDir + "/" + OutputFoldername + "/";
  if (fs::exists(OutputDir)) {
    cerr << "Output directory exists, continuing!" << endl;
  } else {

    cerr << "Output directory does not exist, creating!" << endl;
    if (!fs::create_directory(OutputDir)) {
      fatalError("Unable to create Directory: " + OutputDir);
    }
  }

  // Collect input parameters
  vector<HMInputParamBase *> InParams;

  int numParams = collectInputParams(InParams);
  for (auto param : InParams) {
    cout << "Param: " << *param << "\n";
  }

  // Create json scenario
  string JSonFileNameStr =
      createjson(AppName, OutputFoldername, NumIterations, NumSamples,
                 Predictor, InParams, Objectives);

  // Launch HyperMapper
  string cmd("python3 ");
  cmd += getenv("HYPERMAPPER_HOME");
  cmd += "/scripts/hypermapper.py";
  cmd += " " + JSonFileNameStr;

  cout << "Executing command: " << cmd << endl;
  struct popen2 hypermapper;
  popen2(cmd.c_str(), &hypermapper);

  FILE *instream = fdopen(hypermapper.from_child, "r");
  FILE *outstream = fdopen(hypermapper.to_child, "w");

  const int max_buffer = 1000;
  char buffer[max_buffer];
  // Loop that communicates with HyperMapper
  // Everything is done through function calls,
  // there should be no need to modify bellow this line.
  char* fgets_res;
  int i = 0;
  while (true) {
    fgets_res = fgets(buffer, max_buffer, instream);
    if (fgets_res == NULL) {
      fatalError("'fgets' reported an error.");
    }
    cout << "Iteration: " << i << endl;
    cout << "Recieved: " << buffer;
    // Receiving Num Requests
    string bufferStr(buffer);
    if (!bufferStr.compare("End of HyperMapper\n")) {
      cout << "Hypermapper completed!\n";
      break;
    }
    string NumReqStr = bufferStr.substr(bufferStr.find(' ') + 1);
    int numRequests = stoi(NumReqStr);
    // Receiving input param names
    fgets_res = fgets(buffer, max_buffer, instream);
    if (fgets_res == NULL) {
      fatalError("'fgets' reported an error.");
    }
    bufferStr = string(buffer);
    cout << "Recieved: " << buffer;
    size_t pos = 0;
    // Create mapping for InputParam objects to keep track of order
    map<int, HMInputParamBase *> InputParamsMap;
    string response;
    for (int param = 0; param < numParams; param++) {
      size_t len = bufferStr.find_first_of(",\n", pos) - pos;
      string ParamStr = bufferStr.substr(pos, len);
      //      cout << "  -- param: " << ParamStr << "\n";
      auto paramIt = findHMParamByKey(InParams, ParamStr);
      if (paramIt != InParams.end()) {
        InputParamsMap[param] = *paramIt;
        response += ParamStr;
        response += ",";
      } else {
        fatalError("Unknown parameter received!");
      }
      pos = bufferStr.find_first_of(",\n", pos) + 1;
    }
    for (auto objString : Objectives)
      response += objString + ",";
    if (Predictor)
      response += "Valid";
    response += "\n";
    // For each request
    for (int request = 0; request < numRequests; request++) {
      // Receiving paramter values
      fgets_res = fgets(buffer, max_buffer, instream);
      if (fgets_res == NULL) {
        fatalError("'fgets' reported an error.");
      }
      cout << "Received: " << buffer;
      bufferStr = string(buffer);
      pos = 0;
      for (int param = 0; param < numParams; param++) {
        size_t len = bufferStr.find_first_of(",\n", pos) - pos;
        string ParamValStr = bufferStr.substr(pos, len);
        setInputValue(InputParamsMap[param], ParamValStr);
        response += ParamValStr;
        response += ",";
        pos = bufferStr.find_first_of(",\n", pos) + 1;
      }
      HMObjective Obj = calculateObjective(InParams);
      response += to_string(Obj.f1_value);
      response += ",";
      response += to_string(Obj.f2_value);
      response += ",";
      response += to_string(Obj.valid);
      response += "\n";
    }
    cout << "Response:\n" << response;
    fputs(response.c_str(), outstream);
    fflush(outstream);
    i++;
  }

  deleteInputParams(InParams);
  close(hypermapper.from_child);
  close(hypermapper.to_child);

  FILE *fp;
  string cmdPareto("python3 ");
  cmdPareto += getenv("HYPERMAPPER_HOME");
  cmdPareto += "/scripts/compute_pareto.py";
  cmdPareto += " " + JSonFileNameStr;
  cout << "Executing " << cmdPareto << endl;
  fp = popen(cmdPareto.c_str(), "r");
  while (fgets(buffer, max_buffer, fp))
    printf("%s", buffer);
  pclose(fp);

  return 0;
}

int HMInputParamBase::count = 0;
