#pragma once

#include <stdlib.h>

#ifdef __cplusplus
#include "nn.hpp"
extern "C" {
#endif
  
struct nn_network_t; 
nn_network_t* nn_new_network(size_t* layer_sizes);  

#ifdef __cplusplus
}
#endif
  

