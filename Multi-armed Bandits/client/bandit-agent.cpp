#include <iostream>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <assert.h>
#include <limits>

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#define MAXHOSTNAME 256

using namespace std;


// std::mt19937 gen();
gsl_rng *rng;
std::default_random_engine generator;
const double MACHINE_EPSILON = 1e-8;

void options(){

  cout << "Usage:\n";
  cout << "bandit-agent\n";
  cout << "\t[--numArms numArms]\n";
  cout << "\t[--randomSeed randomSeed]\n";
  cout << "\t[--horizon horizon]\n";
  cout << "\t[--hostname hostname]\n";
  cout << "\t[--port port]\n";
  cout << "\t[--algorithm algorithm]\n";
  cout << "\t[--epsilon epsilon]\n";

}


/*
  Read command line arguments, and set the ones that are passed (the others remain default.)
*/
bool setRunParameters(int argc, char *argv[], int &numArms, int &randomSeed, unsigned long int &horizon, string &hostname, int &port, string &algorithm, double &epsilon){

  int ctr = 1;
  while(ctr < argc){

    //cout << string(argv[ctr]) << "\n";

    if(string(argv[ctr]) == "--help"){
      return false;//This should print options and exit.
    }
    else if(string(argv[ctr]) == "--numArms"){
      if(ctr == (argc - 1)){
	return false;
      }
      numArms = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--randomSeed"){
      if(ctr == (argc - 1)){
	return false;
      }
      randomSeed = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--horizon"){
      if(ctr == (argc - 1)){
	return false;
      }
      horizon = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--hostname"){
      if(ctr == (argc - 1)){
	return false;
      }
      hostname = string(argv[ctr + 1]);
      ctr++;
    }
    else if(string(argv[ctr]) == "--port"){
      if(ctr == (argc - 1)){
	return false;
      }
      port = atoi(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else if(string(argv[ctr]) == "--algorithm"){
      if(ctr == (argc - 1)){
  return false;
      }
      algorithm = string(argv[ctr + 1]);
      ctr++;
    }
     else if(string(argv[ctr]) == "--epsilon"){
      if(ctr == (argc - 1)){
  return false;
      }
      epsilon = atof(string(argv[ctr + 1]).c_str());
      ctr++;
    }
    else{
      return false;
    }

    ctr++;
  }

  return true;
}

/* ============================================================================= */
class banditAgent
{
protected:
  vector<double> estimated_mean;
  vector<int> numPulls;
  int numArms;
  int lastPulled;

  int findMaxIndex(std::vector<double> & values){
    auto begin_iter = values.begin();
    auto end_iter = values.end();
    int index = distance(begin_iter, max_element(begin_iter, end_iter));
    return index;
  }

public:
  banditAgent(){}
  virtual ~banditAgent(){}
  banditAgent(int arms){
    numArms = arms;
    estimated_mean.resize(arms, 0);
    lastPulled = -1;
    numPulls.resize(arms, 0);
  }

  virtual void updateState(float reward){
    if(lastPulled == -1)
      return;
    estimated_mean[lastPulled] = (estimated_mean[lastPulled] * numPulls[lastPulled]
        + reward)/(numPulls[lastPulled] + 1);
    numPulls[lastPulled]++;
  }

  virtual int armToPull(int & pulls) = 0;
};

banditAgent *currAgent;

class epsilonGreedyAgent: public banditAgent{
  private:
    double epsilon;
  public:
    epsilonGreedyAgent(int arms, double eps):banditAgent(arms), epsilon(eps){}
    int armToPull(int & pulls){
      if(gsl_rng_uniform(rng) < epsilon){
        lastPulled = pulls % numArms;
      }
      else{
        lastPulled = findMaxIndex(estimated_mean);
      }
      return lastPulled;
    }
};

class UCBAgent: public banditAgent{
protected:
  std::vector<double> ucb;

  virtual void calc_ucb(int & pulls){
    for (size_t i = 0; i < ucb.size(); ++i){
      ucb[i] = estimated_mean[i] + sqrt(2 * log(pulls)/numPulls[i]);
    }
  }

public:
  UCBAgent(int arms):banditAgent(arms){
    ucb.resize(arms);
  };

  int armToPull(int & pulls){
    if(pulls < numArms){
      lastPulled = pulls;
    }
    else{
      calc_ucb(pulls);
      lastPulled = findMaxIndex(ucb);
    }
    return lastPulled;
  }
};

double KLBernoulli(const double & p, const double & q)
{
  return p * log((p + MACHINE_EPSILON)/q + MACHINE_EPSILON) +
    (1-p)* log((1-p + MACHINE_EPSILON)/(1-q + MACHINE_EPSILON));
}

double generateFromBeta(const int & alpha, const int & beta)
{
    gamma_distribution<double> X(alpha, 1.0);
    gamma_distribution<double> Y(beta, 1.0);
    double x = X(generator);
    return x/(x+ Y(generator) + MACHINE_EPSILON);
}

class KLUCBAgent: public UCBAgent{
private:
  double c;

  void calc_ucb(int & pulls){
    for(size_t i = 0; i < ucb.size(); i++)
      ucb[i] = calcUpperBound(pulls, i);
  }

  double calcUpperBound(int & pulls, int agent)
  {
    double true_mean = estimated_mean[agent];
    double bound = (log(pulls) + c * log(log(pulls)))/numPulls[agent];
    double low, high, mid, kl;
    low = estimated_mean[agent];
    high = 1.0;
    while(true){
      mid = (high + low)/2;
      if(abs(high - mid) < MACHINE_EPSILON)
        break;
      kl = KLBernoulli(true_mean, mid);
      if(kl < bound)
        low = mid;
      else
        high = mid;
    }
    return mid;
  }

public:
  KLUCBAgent(int arms, double cval=3): UCBAgent(arms), c(cval){}
};

class thompsonSamplingAgent: public banditAgent{
private:
  vector<int> sucess;
  vector<int> failures;

  double samplefromBeta(const int & agent) const{
    int alpha = sucess[agent];
    int beta = failures[agent];
    return generateFromBeta(alpha+1, beta+1);
  }

public:
  thompsonSamplingAgent(int numArms):banditAgent(numArms){
    sucess.resize(numArms, 0);
    failures.resize(numArms, 0);
  }

  void updateState(float reward){
    if(lastPulled == -1)
      return;
    if(reward == 0)
      failures[lastPulled]++;
    else if(reward == 1)
      sucess[lastPulled]++;
  }

  int armToPull(int & pulls){
    for (int i = 0; i < numArms; ++i){
      estimated_mean[i] = samplefromBeta(i);
    }
    lastPulled = findMaxIndex(estimated_mean);
    return lastPulled;
  }
};

/* ============================================================================= */

int sampleArm(string algorithm, double epsilon, int pulls, float reward, int numArms, bool firstCall=false)
{
  if(algorithm.compare("rr") == 0){
    return(pulls % numArms);
  }
  if(firstCall){
    assert(pulls == 0);
    if(algorithm.compare("epsilon-greedy") == 0){
      currAgent = new epsilonGreedyAgent(numArms, epsilon);
    }
    else if(algorithm.compare("UCB") == 0){
      currAgent = new UCBAgent(numArms);
    }
    else if(algorithm.compare("KL-UCB") == 0){
      currAgent = new KLUCBAgent(numArms);
    }
    else if(algorithm.compare("Thompson-Sampling") == 0){
      currAgent = new thompsonSamplingAgent(numArms);
    }
    else{
      return -1;
    }
  }
  currAgent->updateState(reward);
  return currAgent->armToPull(pulls);
}


int main(int argc, char *argv[]){
  // Run Parameter defaults.
  int numArms = 5;
  int randomSeed = time(0);
  unsigned long int horizon = 200;
  string hostname = "localhost";
  int port = 5000;
  string algorithm="random";
  double epsilon=0.0;

  //Set from command line, if any.
  if(!(setRunParameters(argc, argv, numArms, randomSeed, horizon, hostname, port, algorithm, epsilon))){
    //Error parsing command line.
    options();
    return 1;
  }

  rng = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(rng, randomSeed);
  generator.seed(randomSeed);


  struct sockaddr_in remoteSocketInfo;
  struct hostent *hPtr;
  int socketHandle;

  bzero(&remoteSocketInfo, sizeof(sockaddr_in));

  if((hPtr = gethostbyname((char*)(hostname.c_str()))) == NULL){
    cerr << "System DNS name resolution not configured properly." << "\n";
    cerr << "Error number: " << ECONNREFUSED << "\n";
    exit(EXIT_FAILURE);
  }

  if((socketHandle = socket(AF_INET, SOCK_STREAM, 0)) < 0){
    close(socketHandle);
    exit(EXIT_FAILURE);
  }

  memcpy((char *)&remoteSocketInfo.sin_addr, hPtr->h_addr, hPtr->h_length);
  remoteSocketInfo.sin_family = AF_INET;
  remoteSocketInfo.sin_port = htons((u_short)port);

  if(connect(socketHandle, (struct sockaddr *)&remoteSocketInfo, sizeof(sockaddr_in)) < 0){
    //code added
    cout<<"connection problem"<<".\n";
    close(socketHandle);
    exit(EXIT_FAILURE);
  }


  char sendBuf[256];
  char recvBuf[256];

  float reward = 0;
  unsigned long int pulls=0;
  int armToPull = sampleArm(algorithm, epsilon, pulls, reward, numArms, true);

  sprintf(sendBuf, "%d", armToPull);

  cout << "Sending action " << armToPull << ".\n";
  while(send(socketHandle, sendBuf, strlen(sendBuf)+1, MSG_NOSIGNAL) >= 0){

    char temp;
    recv(socketHandle, recvBuf, 256, 0);
    sscanf(recvBuf, "%f %c %lu", &reward, &temp, &pulls);
    cout << "Received reward " << reward << ".\n";
    cout<<"Num of  pulls "<<pulls<<".\n";


    armToPull = sampleArm(algorithm, epsilon, pulls, reward, numArms);

    sprintf(sendBuf, "%d", armToPull);
    cout << "Sending action " << armToPull << ".\n";
  }

  close(socketHandle);

  cout << "Terminating.\n";
  delete rng;
  delete currAgent;

  return 0;
}

