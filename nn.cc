#include <nn.h>
#include <vector>
#include <map>
#include <algorithm>
#include <string.h>
#include <iostream>

static inline float activation_f(float x) {
  return x / (1 + abs(x)); 
}

nn_network_t::nn_network_t(size_t outputs, size_t inputs, const std::vector<nn_weight_t>& lst) : nn_network_t(outputs, inputs, &lst[0], lst.size()) {
    
}

nn_network_t::nn_network_t(size_t outputs, size_t inputs, const std::map<size_t, std::map< size_t, float > >& weights) : inputs(inputs), outputs(outputs), weights(weights) {
  size_t mx = outputs + inputs;
  for(auto it = weights.cbegin(); it != weights.cend();it++) {
    auto from = it->first;      
    auto& ws = it->second;
    mx = std::max(mx, from+1);
    for(auto wt = ws.cbegin(); wt != ws.cend(); wt++) {
      mx = std::max(mx, wt->first+1);
    }
  }
  
  activations.resize(mx);
}

nn_network_t::nn_network_t(size_t outputs, size_t inputs, const nn_weight_t* in_w, size_t cnt) : inputs(inputs), outputs(outputs) {
  size_t mx = outputs + inputs;
  for(size_t i = 0;i < cnt;i++) {
    auto& w = in_w[i];
    mx = std::max(mx, w.to);
    weights[w.to][w.from] = w.weight; 
  }
  activations.resize(mx);
}

std::vector<float> nn_network_t::run(const std::vector<float>& input) {
  std::fill(activations.begin(), activations.end(), 0); 
  memcpy(&activations[0], &input[0], inputs * sizeof(float));

  for(auto it = weights.cbegin(); it != weights.cend();it++) {
    auto from = it->first;      
    auto& ws = it->second;

    float val = activations[from];
    if(inputs < from) {
      val = activation_f(val);
      if(val > 0)
	val = 1;
      else
	val = 0;
    }
    for(auto wt = ws.cbegin(); wt != ws.cend(); wt++) {
      auto to = wt->first;
      auto weight = wt->second;
      activations[to] += weight * val;
    }
  }

  std::vector<float> rtn(outputs);
  for(size_t i = 0;i < outputs;i++) {
    rtn[i] = activation_f(activations[inputs + i]);
  }
  return rtn; 
}


