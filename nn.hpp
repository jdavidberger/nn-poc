#include <vector>
#include <map>

struct nn_layer_t {
  std::vector<float> activations; 
  nn_layer_t(size_t s) {}
};

struct nn_links_t {
  std::vector<float> weights;
  nn_links_t(size_t in, size_t out) {
    weights = std::vector<float>(in * out, 1.0); 
  }
};

struct nn_weight_t {
  size_t from, to;
  float weight; 
};
struct nn_network_t {
  size_t outputs, inputs;
  
  std::map<size_t, std::map< size_t, float > > weights; 
  std::vector<float> activations;
  nn_network_t(size_t outputs, size_t inputs, const std::map<size_t, std::map< size_t, float > >& weights);
  nn_network_t(size_t outputs, size_t inputs, const std::vector<nn_weight_t>& lst);
  
  nn_network_t(size_t outputs, size_t inputs, const nn_weight_t* in_w, size_t cnt); 

  std::vector<float> run(const std::vector<float>& input);   
};

