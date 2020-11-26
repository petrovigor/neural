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

//activation function
float activation_fn(float val) {
  //return bounds [0.f, 1.f]
  return 1.f / ( 1.f + std::exp( -1.f * val ) );
}

class NeuralNetwork {
public:
  //network should be initialized by passing two dimensional array
  //of float values, where values doesn't really matter,
  //matters only count of elements
  //
  //example of network with 3 layers(input + hidden + output):
  // { 1.f, 0.f },           <- two input signals
  // { 0.f, 0.f, 0.f, 0.f }, <- four neurons in hidden layer
  // { 0.f },                <- only one output signal
  NeuralNetwork(const std::vector<std::vector<node>>& vv) {
    nodes = vv;
    buildStartupWeights();
  }

  void print() const {
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

  //travers neural network and updates node values each once
  void traverse() {
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
  }

  float calculateError() {
    //last node of the last layer is output node
    const node& output_node = nodes.back( ).back( );
    return 1.f - output_node.val;
  }

private:
  //called once by ctor; allocates weights of all links
  //between all neurons in the network
  //keeps them in one-dimensional array
  void buildStartupWeights() {
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
  }

private:
  std::vector<std::vector<node>> nodes;
  std::vector<float> weights;

};

int main() {
  NeuralNetwork nn(std::vector<std::vector<node>> {
    { 0, 0 },
    { 0, 0, 0, 0 },
    { 0 }
  });

  nn.print();

  nn.traverse();

  //neural network hello world program implements
  //xor function, where we have two input signals,
  //one hidden layer with 4 nodes
  //and only one output signal

  //when input signals are '1' and '0' for xor fn (1 ^ 0),
  //we are waiting for 1 output, so
  //error for the last node will be defined as: '1.f - x'

  std::cout << "first generation finished with error: " <<
    nn.calculateError() << std::endl;

  nn.print();

  return 0;
}
