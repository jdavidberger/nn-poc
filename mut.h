#pragma once

#include <iostream>
#include <apply.h>
#include <nn.h>
#include <random>
#include <algorithm>

struct nn_env {
  const size_t IN;
  const size_t OUT; 

    int population_size = 1000;
    float keep   = 0.01; 
    float mutate_p = 0.90;
    float cross  = 0.00;
    
    struct genome {
      size_t max_size; 
      std::map<size_t, std::map<size_t, float> > weights;
      bool operator<(const genome& g) const {
	return max_size < g.max_size; 
      }
    };

    typedef std::vector<genome> samples_t;
    typedef std::vector<std::pair<float, genome>> weightedSamples_t;
    
    std::default_random_engine generator;
    std::normal_distribution<float> distribution;
    std::poisson_distribution<int>  pdistribution;

    virtual float eval(const genome& g) = 0;
    virtual bool isDone(const weightedSamples_t& s) { return false; }
  
nn_env(size_t in, size_t out) : distribution(0.0,0.5), pdistribution(3), IN(in), OUT(out) {}
    
    int train() {
      std::vector<genome> population(population_size);

      for(size_t i = 0;i < population_size;i++) {
	population.emplace_back( random() ); 
      }
      
      size_t rounds = 0; 
      weightedSamples_t ws(population.size());
      do {
	for(size_t i = 0;i < population.size();i++) {
	  float e = eval(population[i]); 
	  ws[i] = std::make_pair(e, population[i]);
	}

	std::sort(ws.begin(), ws.end());
	std::reverse(ws.begin(), ws.end());
	rounds++;
	
	std::cout << rounds << " Max: " << ws[0].first << ", " << ws[0].second.max_size << std::endl; 

	
	size_t keep_n = population.size() * keep; 
	for(size_t i = 0;i < keep_n;i++) {
	  population[i] = ws[i].second;
	}

	size_t resample_mutate_n = population.size() * mutate_p; 
	resample(ws, &population[keep_n], resample_mutate_n); 
	for(size_t i = keep_n; i < resample_mutate_n + keep_n;i++) {
	  mutate(population[i]); 
	}
	
      } while(isDone(ws) == false);
      return rounds;
    }
    
    void  mutate(genome& ind) {
      int new_idx = -1; 
      if(rand() < RAND_MAX / 100) {
	new_idx = ind.max_size + IN + OUT; 
	ind.max_size++;

	ind.weights[ rand() * IN / static_cast <float> (RAND_MAX) ][new_idx] = distribution(generator);
	ind.weights[ new_idx ][ IN + (rand() * OUT / static_cast <float> (RAND_MAX)) ] = distribution(generator);
      }
      size_t mutations = pdistribution(generator); 
      for(size_t m = 0;m < mutations;m++) {
	size_t i = rand() * (IN + OUT + ind.max_size) / static_cast <float> (RAND_MAX);
	size_t j = rand() * (IN + OUT + ind.max_size) / static_cast <float> (RAND_MAX);
	bool iIsInput = i < IN, jIsInput = j < IN;
	bool iIsOutput = !iIsInput && i < (IN + OUT), jIsOutput = !jIsInput && j < (IN + OUT);
	if(iIsOutput && jIsOutput || iIsInput && jIsInput)
	  break;
	ind.weights[i][j] += distribution(generator) / 5.0; 
      }
    }


    genome random(size_t s = 0) {
      genome rtn;
      rtn.max_size = s;
      for(size_t i = 0;i < IN;i++) {
	for(size_t j = IN;j < IN + OUT + rtn.max_size;j++) {
	  rtn.weights[i][j] = distribution(generator);
	}
      }

      for(size_t i = 0;i < s;i++) {
	for(size_t j = 0; j < OUT;j++) {
	  rtn.weights[i + IN + OUT][j + IN] = distribution(generator);
	}
	for(size_t j = i + 1;j < s;j++) {
	  rtn.weights[i + IN + OUT][j + IN + OUT] = distribution(generator);
	}
      }
  
      return rtn; 
    }

    void resample(std::vector<std::pair<float, genome> >& pop, genome* output, size_t pop_size) {
      float mx = pop[0].first, mn = pop[0].first;
      size_t ms = 0; 
      for(auto& p : pop) {
	ms = std::max(ms, p.second.max_size); 
	mx = std::max(mx, p.first);
	mn = std::min(mn, p.first);
      }
      std::vector<float> cdf; cdf.reserve(pop.size());
      float cumm = 0, rng = 0; 
      for(auto& p : pop) {
	cumm += (p.first - mn + 0.01) * (p.first - mn + 0.01);
	cdf.push_back(cumm);
      }
      rng = cumm; 

      for(size_t i = 0;i < pop_size;i++) {
	auto t = rand() * rng / static_cast <float> (RAND_MAX);; 
	auto pick = std::lower_bound(cdf.begin(), cdf.end(), t) - cdf.begin();
	if(pick >= pop.size()) pick = pop.size() - 1;
	output[i] = pop[pick].second;
      }
    }

  };
