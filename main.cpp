#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <time.h>

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
  return 1.f / ( 1.f + std::exp( -val ) );
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

    //there can be no hidden layers
    //only input + output
    if(layers_count < 2)
      throw std::invalid_argument("It should be at least two layers");

    layers.reserve( layers_count );
    for(size_t i = 0; i < layers_count; ++i) {
      if( !layer_sizes[i] )
        throw std::invalid_argument("Layer should not be empty");

      layers.emplace_back( std::vector<node>( layer_sizes[i] ) );
    }

    input_neurons = layers.front( ).size( );
    output_neurons = layers.back( ).size( );

    buildStartupWeights();
    learnCoefficent = 0.8f;
    generations = 0;

    std::cout << "Created neural network with " <<
      layers_count << " layers, " <<
      input_neurons << " input signals, " <<
      output_neurons << " output signals and " <<
      weights.size() << " weights" << std::endl;
  }

  void print_values() const {
    std::cout << "neuron values: " << std::endl;

    size_t max_lines = 0;
    for(size_t i = 0; i < layers.size(); ++i)
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

    std::cout << std::endl;
  }

  void print_weights() const {
    std::cout << "weights: " << std::endl;

    for(size_t i = 1; i < layers.size(); ++i) {
      const size_t n_prev = layers[i - 1].size();
      const size_t n_next = layers[i].size();

      size_t weight_index = 0;

      for(size_t k = 0; k < n_next; ++k) {
        for(size_t j = 0; j < n_prev; ++j) {
          std::cout << weights[weight_index + j] << " ";
        }
        std::cout << std::endl;

        weight_index += n_prev;
      }
    }
  }

  //travers neural network and updates node values each once
  void traverse() {
    for(size_t i = 1; i < layers.size(); ++i) {
#if 0
      std::cout << "count " << i << " layer of " << (layers.size() - 1) << std::endl;
#endif //0

      const size_t n_prev = layers[i - 1].size();
      const size_t n_next = layers[i].size();

      size_t weight_index = 0;

      for(size_t k = 0; k < n_next; ++k) {
        float val = 0.f;

#if 0
        std::cout << "[" << layers[i].nodes[k].index << "] = f(";
#endif //0

        //std::cout << "from " << nodes[i - 1][j] << " to " << nodes[i][k] << std::endl;

        for(size_t j = 0; j < n_prev; ++j) {
          val += ( layers[i - 1].nodes[j].val * weights[weight_index + j] );
#if 0
          std::cout << "(" << layers[i - 1].nodes[j].val << " * " << weights[weight_index + j];
          if(j < n_prev - 1)
            std::cout << ") + ";
          else
            std::cout << "))";
#endif //0
        }
#if 0
        std::cout << std::endl;
#endif //0

        val = activation_fn(val);

        layers[i].nodes[k].val = val;

        weight_index += n_prev;
      }
    }
  }

  auto getGenerations() const noexcept {
    return generations;
  }

  void nextGeneration() {
    ++generations;
  }

  void calculateErrors() {
    //calculate error for output layer
    Layer& outLayer = getOutputLayer( );
    const auto& targetLayer = getTargetLayer( );

    const size_t outNodes = outLayer.size( );

#if 0
    std::cout << "output layer errors: ";
#endif //0

    float outputLayerError = 0.f;

    for(size_t i = 0; i < outNodes; ++i) {
      outLayer.nodes[i].error = targetLayer[i] - outLayer.nodes[i].val;

      outputLayerError += outLayer.nodes[i].error;

#if 0
      std::cout << targetLayer[i] << " - " << outLayer.nodes[i].val <<
        " = " << outLayer.nodes[i].error << " ";
#endif //0
    }

    outputLayerError /= outNodes;
    std::cout << "err = " <<  outputLayerError << std::endl;

#if 0
    std::cout << std::endl;
#endif //0

    //node& output_node = outputLayer().nodes.back( );
    //output_node.error = 1.f - output_node.val;
#if 1
    //std::cout << "finding errors for output node: " << output_node.error << std::endl;
#endif //0

    //calculate errors for hidden layers
    for(size_t i = layers.size() - 1; i > 1; --i) {
#if 0
      std::cout << "finding errors for " << i << " layer" << std::endl;
#endif //0

      auto& hidden = layers[i - 1]; //hidden layer
      const auto& output = layers[i]; //output layer

      //after that weight index points to the weight
      //between the first node of hidden and node of output(or next) layer
      size_t weight_index = hidden.weight_start_index;

      for(size_t j = 0; j < hidden.size(); ++j) {
        //error for each node of hidden layer
        float error = 0.f;

#if 0
        std::cout << "\t" << "errors for node " << hidden.nodes[j].index << " windex: " << weight_index;
#endif //0

        for(size_t k = 0; k < output.size(); ++k) {
          const float temp_error = output.nodes[k].error * weights[weight_index];
#if 0
          std::cout << "temp_error: " << temp_error << std::endl;
#endif //0

          hidden.nodes[j].error += temp_error;
        }

#if 0
        std::cout << " " << error << std::endl;
#endif //0

        //move weight_index to the weight
        //of the next node in hidden layer
        ++weight_index;
      }
    }
  }

  void setSourceData(const std::vector<float>& newInput) {
    auto& input = getInputLayer( );

    if(input.size() != newInput.size()) {
      throw std::invalid_argument("Bad values for input layer");
    }

    size_t i = 0;
    for(auto& node : input.nodes) {
      node.val = newInput[i++];
    }
  }

  void setTargetData(const std::vector<float>& newValues) {
    auto& target = getTargetLayer( );

    const size_t n = target.size();

    if(n != newValues.size()) {
      throw std::invalid_argument("Bad values for target layer");
    }

    for(size_t i = 0; i < n; ++i) {
      target[i] = newValues[i];
    }
  }

  //updates weights between neurons based on thier error
  void learn() {
#if 0
    std::cout << "learn" << std::endl;
#endif //0

    size_t w_index = 0;

    for(size_t i = 1; i < layers.size(); ++i) {
#if 0
      std::cout << "updating weights between " << (i-1) << " & " << i << std::endl;
#endif //0

      const size_t from = layers[i - 1].size( );
      const size_t to = layers[i].size( );

      for(size_t j = 0; j < from; ++j) {
        for(size_t k = 0; k < to; ++k) {

          weights[w_index] = weights[w_index] - learnCoefficent *
            activation_fn( layers[i - 1].nodes[j].error ) *
            ( 1.f - activation_fn( layers[i - 1].nodes[j].error ) ) *
            layers[i].nodes[k].val;

#if 0
          std::cout << "from " << layers[i - 1].nodes[j].index <<
            " to " << layers[i].nodes[k].index <<
            " index = " << w_index << std::endl;
#endif //0

          ++w_index;

        }
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

    const auto& outputLayer = getOutputLayer( );
    targetLayer.resize( outputLayer.size( ) );
  }

  Layer& getInputLayer() {
    return layers.front( );
  }

  Layer& getOutputLayer() {
    return layers.back( );
  }

  std::vector<float>& getTargetLayer() {
    return targetLayer;
  }

  const std::vector<float>& getTargetLayer() const {
    return targetLayer;
  }

private:
  std::vector<Layer> layers;
  std::vector<float> weights;
  std::vector<float> targetLayer; //contains target data
  size_t input_neurons;
  size_t output_neurons;
  size_t generations;
  float learnCoefficent; //[0.f, 1.f]

};

struct Teacher {

  struct TestingCase {
    std::vector<float> source;
    std::vector<float> target;

    TestingCase(const std::vector<float>& in, const std::vector<float>& out)
      : source(in)
      , target(out)
    { }
  };

  void addTestingCase(const std::vector<float>& in, const std::vector<float>& out) {
    tests.emplace_back( TestingCase(in, out) );
  }

  void teach(NeuralNetwork& nn, int gensToTeach) {
    const size_t tests_total = tests.size();

    std::cout << "Teaching " << gensToTeach << " generations total" << std::endl;

    if(!tests_total) {
      std::cerr << "No test cases found. Teaching finished" << std::endl;
      return;
    }

    do {
      nn.nextGeneration();

      for(size_t i = 0; i < tests_total; ++i) {
#if 0
        std::cout << "Test case #" << i << std::endl;
#endif //0

        nn.setSourceData( tests[i].source );
        nn.setTargetData( tests[i].target );

        nn.traverse( );
        nn.calculateErrors();
        nn.learn();
      }

#if 0
      std::cout << "Gen: " << nn.getGenerations() << " error = " << lastError << std::endl;
#endif //0

    } while( --gensToTeach > 0 );

    std::cout << "No more generations to teach" << std::endl;
  }

  std::vector<TestingCase> tests;

};

int main() {
  std::srand( time( 0 ) );

  NeuralNetwork nn ({ 1, 1 });

  Teacher teacher;
  teacher.addTestingCase( { 1.f }, { 0.25f } );

  teacher.teach( nn, 25 );

  return 0;
}
