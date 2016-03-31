#include <iostream>
#include <apply.h>
#include <mut.h>
#include <random>

struct xor_env : public nn_env  {
  typedef typename nn_env::genome genome;
  typedef typename nn_env::weightedSamples_t weightedSamples_t;
  
  virtual float eval(const genome& g) override {
    nn_network_t nn(OUT, IN, g.weights);
    double rtn = 0; 
    for(int i = 0;i < pow(2, IN - 1);i++) {
      int j = i; 
      std::vector<float> input = { 1. };
      bool ans = false; 
      for(int z = 0;z < IN - 1;z++) {
	bool flag = j % 2 == 1;
	j = j >> 1;
	ans = ans ^ flag; 
	input.push_back(flag); 
      }
    
      auto output = nn.run(input);
      bool guess = output[0] > 0;
      rtn += (-std::fabs(output[0] - (ans ? 1.0 : -1.0))) + (ans == guess ? 101 : 0);
    }
    return rtn;
  }

  virtual bool isDone(const weightedSamples_t& s) {
    for(auto& ans : s) {
      if(ans.first >= pow(2, IN - 1) * 100) {
	return true;
      }
    }
    return false; 
  }


  xor_env(size_t IN) : nn_env(IN + 1, 1) {}
};

int main(int argc, char* argv[])
{
  for(size_t i = 2;i < 6;i++) {
    xor_env env(i);
    auto solvedIn = env.train();
    std::cout << "Solved " << i << " in " << solvedIn << " rounds" << std::endl;
  }
  return 0;    
}

