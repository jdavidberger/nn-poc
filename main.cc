#include <nn.h>
#include <iostream>
#include <mut.h>

int main() {
  std::map<size_t, std::map<size_t, float> > weights;

  weights[0][3] =  -1.5;
  weights[0][4] =  -0.5;
  weights[0][5] =   1.5;
  
  weights[1][4] =   1;
  weights[1][5] =  -1;
  
  weights[2][4] =  1; 
  weights[2][5] = -1;

  weights[4][3] = 1;
  weights[5][3] = 1;
  
  nn_network_t nn(1, 3, weights);

  for(int i = 0;i < 4;i++) {
    bool a = i % 2 == 0; 
    bool b = (i >> 1) % 2 == 0;
    bool c = a ^ b;

    auto out = nn.run({1, a ? 1.0 : 0.0, b ? 1.0 : 0.0});
    std::cout << a << ", " << b << ", " << c << ", " << out[0] << std::endl;
  }

  return 0;
}
