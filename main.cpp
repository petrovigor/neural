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
  node()
   : val(0.f)
   , index(++node_index)
   , error(0.f)
  { }

  int index; //only for debug print
  float val; //its not a weight
  float error;
};

//activation function
float activation_fn(float val) {
  //return bounds [0.f, 1.f]
  return 1.f / ( 1.f + std::exp( -1.f * val ) );
}

struct Layer {
  Layer(const std::vector<node>& _nodes)
    : nodes(_nodes)
    , weight_start_index(0u)
  { }

  size_t size() const noexcept {
    return nodes.size();
  }

  void setWeightStartIndex(size_t i) {
    weight_start_index = i;
  }

  std::vector<node> nodes;

  //this logic should be moved into main class
  size_t weight_start_index;
};

class NeuralNetwork {
public:
  //network should be initialized by passing size_t array
  //of layers represented

  //NeuralNetwork nn ({ 2, 4, 1 });
  //creates network with three layers:
  // first (input layer) with two signals,
  // second (first hidden layer) with four neurons and
  // third (output layer) with one signal
  NeuralNetwork(const std::vector<int>& layer_sizes) {
    const size_t layers_count = layer_sizes.size();
    if(layers_count < 3)
      throw std::invalid_argument("It should be at least three layers");

    layers.reserve( layers_count );
    for(size_t i = 0; i < layers_count; ++i) {
      if( !layer_sizes[i] )
        throw std::invalid_argument("Layer should not be empty");

      layers.emplace_back( std::vector<node>( layer_sizes[i] ) );
    }

    input_neurons = layers.front( ).size( );
    output_neurons = layers.back( ).size( );

    buildStartupWeights();

    std::cout << "Created neural network with " <<
      layers_count << " layers, " <<
      input_neurons << " input signals, " <<
      output_neurons << " output signals and " <<
      weights.size() << " weights" << std::endl;
  }

  void print() const {
    size_t max_lines = 0;
    for(int i = 0; i < layers.size(); ++i)
      max_lines = std::max(max_lines, layers[i].size());

    for(size_t i = 0; i < max_lines; ++i) {
      for(size_t j = 0; j < layers.size(); ++j) {
        if(layers[j].size() > i)
          std::cout << std::to_string(layers[j].nodes[i].val) << "\t";
        else
          std::cout << "        " << "\t";
      }
      std::cout << std::endl;
    }
  }

  //travers neural network and updates node values each once
  void traverse() {
    for(int i = 1; i < layers.size(); ++i) {
      std::cout << "count " << i << " layer of " << (layers.size() - 1) << std::endl;

      const size_t n_prev = layers[i - 1].size();
      const size_t n_next = layers[i].size();

      size_t weight_index = 0;

      for(int k = 0; k < n_next; ++k) {
        float val = 0.f;

        std::cout << "[" << layers[i].nodes[k].index << "] = f(";

        //std::cout << "from " << nodes[i - 1][j] << " to " << nodes[i][k] << std::endl;

        for(int j = 0; j < n_prev; ++j) {
          val += ( layers[i - 1].nodes[j].val * weights[weight_index + j] );
          std::cout << "(" << layers[i - 1].nodes[j].val << " * " << weights[weight_index + j];
          if(j < n_prev - 1)
            std::cout << ") + ";
          else
            std::cout << "))";
        }
        std::cout << std::endl;

        val = activation_fn(val);

        layers[i].nodes[k].val = val;

        weight_index += n_prev;
      }
    }
  }

  void calculateErrors() {
    //calculate error for output node
    node& output_node = outputLayer().nodes.back( );
    output_node.error = 1.f - output_node.val;
    std::cout << "finding errors for output node: " << output_node.error << std::endl;

    //calculate errors for hidden layers
    for(size_t i = layers.size() - 1; i > 1; --i) {
      std::cout << "finding errors for " << i << " layer" << std::endl;

      auto& hidden = layers[i - 1]; //hidden layer
      const auto& output = layers[i]; //output layer

      //after that weight index points to the weight
      //between the first node of hidden and node of output(or next) layer
      size_t weight_index = hidden.weight_start_index;

      for(size_t j = 0; j < hidden.size(); ++j) {
        float error = 0.f;

        std::cout << "\t" << "errors for node " << hidden.nodes[j].index << " windex: " << weight_index;

        for(size_t k = 0; k < output.size(); ++k) {
          error += output.nodes[k].error * weights[weight_index];
        }

        hidden.nodes[j].error = error;
        std::cout << " " << error << std::endl;

        //move weight_index to the weight
        //of the next node in hidden layer
        ++weight_index;
      }
    }
  }

private:
  //called once by ctor; allocates weights of all links
  //between all neurons in the network
  //keeps them in one-dimensional array
  void buildStartupWeights() {
    size_t weight_n = 0;

    //get count of weights between neurons
    for(size_t i = 1; i < layers.size(); ++i) {
      const size_t n_prev = layers[i - 1].size();
      const size_t n_next = layers[i].size();

      layers[i - 1].setWeightStartIndex(weight_n);

#if _DEBUG && 0
      std::cout << "start index of layer[" << (i-1) << "] is " <<
        layers[i - 1].weight_start_index << std::endl;
#endif //DEBUG

      weight_n += n_prev * n_next;
    }

    weights.reserve(weight_n);

    for(size_t i = 0; i < weight_n; ++i) {
      //fill weights to random low value
      weights.emplace_back( randw() );
    }
  }

  Layer& inputLayer() {
    return layers.front( );
  }

  Layer& outputLayer() {
    return layers.back( );
  }

private:
  std::vector<Layer> layers;
  std::vector<float> weights;
  size_t input_neurons;
  size_t output_neurons;

};

int main() {
  NeuralNetwork nn ({ 2, 4, 1 });

  nn.print();

  nn.traverse();

  //neural network hello world program implements
  //xor function, where we have two input signals,
  //one hidden layer with 4 nodes
  //and only one output signal

  //when input signals are '1' and '0' for xor fn (1 ^ 0),
  //we are waiting for 1 output, so
  //error for the last node will be defined as: '1.f - x'

  nn.calculateErrors();

  return 0;
}
