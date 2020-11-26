#include <vector>
#include <iostream>
#include <algorithm>
#include <string>

float randf() {
  //returns [0.0f, 1.0f]
  return static_cast<float>( ( rand() % 101 ) / 100.f );
}

float randw() {
  //returns [0.0f, 0.35f]
  return static_cast<float>( ( rand() % 36 ) / 100.f );
}

static int node_index = 0;
struct node {
  node(float _v)
   : val(_v)
   , index(++node_index)
  { }

  int index; //only for debug print
  float val; //its not a weight
};

void print(const std::vector<std::vector<node>>& nodes) {
  size_t max_lines = 0;
  for(int i = 0; i < nodes.size(); ++i)
    max_lines = std::max(max_lines, nodes[i].size());

  for(size_t i = 0; i < max_lines; ++i) {
    for(size_t j = 0; j < nodes.size(); ++j) {
      if(nodes[j].size() > i)
        std::cout << std::to_string(nodes[j][i].val) << "\t";
      else
        std::cout << "        " << "\t";
    }
    std::cout << std::endl;
  }
}

//activation function
float activation_fn(float val) {
  //return bounds [0.f, 1.f]
  return 1.f / ( 1.f + std::exp( -1.f * val ) );
}

int main() {

  std::vector<std::vector<node>> nodes {
    { 1.f, 0.f },
    { 0.f, 0.f, 0.f, 0.f }, //start values of any hidden layer is whatever
    { 0.f },
  };

  std::vector<float> weights; //may be it should be vector<vector<float>>
  size_t weight_n = 0;

  //get count of weights between neurons
  for(int i = 1; i < nodes.size(); ++i) {
    const size_t n_prev = nodes[i - 1].size();
    const size_t n_next = nodes[i].size();

    weight_n += n_prev * n_next;
  }

  weights.reserve(weight_n);

  std::cout << "weights count: " << weight_n << std::endl;
  for(size_t i = 0; i < weight_n; ++i) {
    //fill weights to random low value
    weights.emplace_back( randw() );
  }

  print(nodes);

  // 1 = input layer
  // 2 = hidden layer
  // 3 = output layer
  //
  //   1  2  3
  // ------------
  //   1  3  7
  //   2  4
  //      5
  //      6

  //travers neural network
  for(int i = 1; i < nodes.size(); ++i) {
    std::cout << "count " << i << " layer of " << (nodes.size() - 1) << std::endl;

    const size_t n_prev = nodes[i - 1].size();
    const size_t n_next = nodes[i].size();
    
    size_t weight_index = 0;
    
    for(int k = 0; k < n_next; ++k) {
      float val = 0.f;

      std::cout << "[" << nodes[i][k].index << "] = f(";

      //std::cout << "from " << nodes[i - 1][j] << " to " << nodes[i][k] << std::endl;

      for(int j = 0; j < n_prev; ++j) {
        val += ( nodes[i - 1][j].val * weights[weight_index + j] );
        std::cout << "(" << nodes[i - 1][j].val << " * " << weights[weight_index + j];
        if(j < n_prev - 1)
          std::cout << ") + ";
        else
          std::cout << "))";
      }
      std::cout << std::endl;

      val = activation_fn(val);

      nodes[i][k].val = val;

      weight_index += n_prev;
    }
  }

  std::cout << "first generation: " << std::endl;
  print(nodes);

  return 0;
}
