#ifndef NETWORK_H
#define NETWORK_H

#include "neuralClasses.h"
#include "util.h"

namespace nplm
{

// is this cheating?
using Eigen::Matrix;
using Eigen::MatrixBase;
using Eigen::Dynamic;

class propagator {
  int minibatch_size;
  model *pnn;

 public:
  Node<Input_word_embeddings> input_layer_node;
  Node<Linear_layer> first_hidden_linear_node;
  Node<Activation_function> first_hidden_activation_node;
  Node<Linear_layer> second_hidden_linear_node;
  Node<Activation_function> second_hidden_activation_node;
  Node<Output_word_embeddings> output_layer_node;
  bool skip_hidden;

 public:
  propagator () : minibatch_size(0), pnn(0) { }

  propagator (model &nn, int minibatch_size)
      :
      pnn(&nn),
      input_layer_node(&nn.input_layer, minibatch_size),
      first_hidden_linear_node(&nn.first_hidden_linear, minibatch_size),
      first_hidden_activation_node(&nn.first_hidden_activation, minibatch_size),
      second_hidden_linear_node(&nn.second_hidden_linear, minibatch_size),
      second_hidden_activation_node(&nn.second_hidden_activation, minibatch_size),
      output_layer_node(&nn.output_layer, minibatch_size),
      minibatch_size(minibatch_size)
  {
    skip_hidden = (nn.num_hidden == 0);
  }

  // This must be called if the underlying model is resized.
  void resize(int minibatch_size) {
    this->minibatch_size = minibatch_size;
    input_layer_node.resize(minibatch_size);
    first_hidden_linear_node.resize(minibatch_size);
    first_hidden_activation_node.resize(minibatch_size);
    second_hidden_linear_node.resize(minibatch_size);
    second_hidden_activation_node.resize(minibatch_size);
    output_layer_node.resize(minibatch_size);
  }

  void resize() { resize(minibatch_size); }

  template <typename Derived>
  void fProp(const MatrixBase<Derived> &data)
  {
    if (!pnn->premultiplied)
    {
      start_timer(0);
      input_layer_node.param->fProp(data, input_layer_node.fProp_matrix);
      stop_timer(0);

      start_timer(1);
      first_hidden_linear_node.param->fProp(input_layer_node.fProp_matrix,
                                            first_hidden_linear_node.fProp_matrix);
    }
    else
    {
      int n_inputs = first_hidden_linear_node.param->n_inputs();
      USCMatrix<double> sparse_data;
      input_layer_node.param->munge(data, sparse_data);

      start_timer(1);
      first_hidden_linear_node.param->fProp(sparse_data,
                                            first_hidden_linear_node.fProp_matrix);
    }
    first_hidden_activation_node.param->fProp(first_hidden_linear_node.fProp_matrix,
                                              first_hidden_activation_node.fProp_matrix);
    //std::cerr<<"in fprop first hidden activation node fprop is "<<first_hidden_activation_node.fProp_matrix<<std::endl;
    //std::getchar();
    stop_timer(1);


    if (!skip_hidden) {
      start_timer(2);
      second_hidden_linear_node.param->fProp(first_hidden_activation_node.fProp_matrix,
                                             second_hidden_linear_node.fProp_matrix);
      second_hidden_activation_node.param->fProp(second_hidden_linear_node.fProp_matrix,
                                                 second_hidden_activation_node.fProp_matrix);
      stop_timer(2);
    }

    // The propagation stops here because the last layer is very expensive.
  }

  // Dense version (for standard log-likelihood)
  template <typename DerivedIn, typename DerivedOut>
  void bProp(const MatrixBase<DerivedIn> &data,
             const MatrixBase<DerivedOut> &output,
             double learning_rate,
             double momentum,
             double L2_reg,
             std::string &parameter_update,
             double conditioning_constant,
             double decay)
  {
    // Output embedding layer

    start_timer(7);
    output_layer_node.param->bProp(output,
                                   output_layer_node.bProp_matrix);
    stop_timer(7);

    start_timer(8);
    Node<Activation_function> & final_hidden_activation_node = skip_hidden ? first_hidden_activation_node : second_hidden_activation_node;
    if (parameter_update == "SGD") {
      output_layer_node.param->computeGradient(final_hidden_activation_node.fProp_matrix,
                                               output,
                                               learning_rate,
                                               momentum);
    } else if (parameter_update == "ADA") {
      output_layer_node.param->computeGradientAdagrad(final_hidden_activation_node.fProp_matrix,
                                                      output,
                                                      learning_rate);
    } else if (parameter_update == "ADAD") {
      //std::cerr<<"Adadelta gradient"<<endl;
      int current_minibatch_size = final_hidden_activation_node.fProp_matrix.cols();
      output_layer_node.param->computeGradientAdadelta(final_hidden_activation_node.fProp_matrix,
                                                       output,
                                                       1.0/current_minibatch_size,
                                                       conditioning_constant,
                                                       decay);
    } else {
      std::cerr<<"Parameter update :"<<parameter_update<<" is unrecognized"<<std::endl;
    }
    stop_timer(8);

    bPropRest(data,
              learning_rate,
              momentum,
              L2_reg,
              parameter_update,
              conditioning_constant,
              decay);
  }

  // Sparse version (for NCE log-likelihood)
  template <typename DerivedIn, typename DerivedOutI, typename DerivedOutV>
  void bProp(const MatrixBase<DerivedIn> &data,
             const MatrixBase<DerivedOutI> &samples,
             const MatrixBase<DerivedOutV> &weights,
             double learning_rate,
             double momentum,
             double L2_reg,
             std::string &parameter_update,
             double conditioning_constant,
             double decay)
  {

    // Output embedding layer

    start_timer(7);
    output_layer_node.param->bProp(samples,
                                   weights,
                                   output_layer_node.bProp_matrix);
    stop_timer(7);


    start_timer(8);
    Node<Activation_function> & final_hidden_activation_node = skip_hidden ? first_hidden_activation_node : second_hidden_activation_node;
    if (parameter_update == "SGD") {
      output_layer_node.param->computeGradient(final_hidden_activation_node.fProp_matrix,
                                               samples,
                                               weights,
                                               learning_rate,
                                               momentum);
    } else if (parameter_update == "ADA") {
      output_layer_node.param->computeGradientAdagrad(final_hidden_activation_node.fProp_matrix,
                                                      samples,
                                                      weights,
                                                      learning_rate);
    } else if (parameter_update == "ADAD") {
      int current_minibatch_size = final_hidden_activation_node.fProp_matrix.cols();
      //std::cerr<<"Adadelta gradient"<<endl;
      output_layer_node.param->computeGradientAdadelta(final_hidden_activation_node.fProp_matrix,
                                                       samples,
                                                       weights,
                                                       1.0/current_minibatch_size,
                                                       conditioning_constant,
                                                       decay);
    } else {
      std::cerr<<"Parameter update :"<<parameter_update<<" is unrecognized"<<std::endl;
    }

    stop_timer(8);

    bPropRest(data,
              learning_rate,
              momentum,
              L2_reg,
              parameter_update,
              conditioning_constant,
              decay);
  }

 private:
  template <typename DerivedIn>
  void bPropRest(const MatrixBase<DerivedIn> &data,
                 double learning_rate, double momentum, double L2_reg,
                 std::string &parameter_update,
                 double conditioning_constant,
                 double decay)
  {
    // Second hidden layer



    // All the compute gradient functions are together and the backprop
    // functions are together
    ////////BACKPROP////////////
    start_timer(9);
    if (skip_hidden)
    {
      start_timer(9);
      first_hidden_activation_node.param->bProp(output_layer_node.bProp_matrix,
                                                first_hidden_activation_node.bProp_matrix,
                                                first_hidden_linear_node.fProp_matrix,
                                                first_hidden_activation_node.fProp_matrix);

      first_hidden_linear_node.param->bProp(first_hidden_activation_node.bProp_matrix,
                                            first_hidden_linear_node.bProp_matrix);
      stop_timer(9);

    }
    else
    {
      second_hidden_activation_node.param->bProp(output_layer_node.bProp_matrix,
                                                 second_hidden_activation_node.bProp_matrix,
                                                 second_hidden_linear_node.fProp_matrix,
                                                 second_hidden_activation_node.fProp_matrix);


      second_hidden_linear_node.param->bProp(second_hidden_activation_node.bProp_matrix,
                                             second_hidden_linear_node.bProp_matrix);
      stop_timer(9);

      start_timer(11);
      first_hidden_activation_node.param->bProp(second_hidden_linear_node.bProp_matrix,
                                                first_hidden_activation_node.bProp_matrix,
                                                first_hidden_linear_node.fProp_matrix,
                                                first_hidden_activation_node.fProp_matrix);

      first_hidden_linear_node.param->bProp(first_hidden_activation_node.bProp_matrix,
                                            first_hidden_linear_node.bProp_matrix);
      stop_timer(11);
    }
    //std::cerr<<"First hidden layer node backprop matrix is"<<first_hidden_linear_node.bProp_matrix<<std::endl;
    //std::getchar();
    ////COMPUTE GRADIENT/////////
    if (parameter_update == "SGD") {
      if (!skip_hidden)
      {
        start_timer(10);
        second_hidden_linear_node.param->computeGradient(second_hidden_activation_node.bProp_matrix,
                                                         first_hidden_activation_node.fProp_matrix,
                                                         learning_rate,
                                                         momentum,
                                                         L2_reg);
        stop_timer(10);
      }

      // First hidden layer


      start_timer(12);
      first_hidden_linear_node.param->computeGradient(first_hidden_activation_node.bProp_matrix,
                                                      input_layer_node.fProp_matrix,
                                                      learning_rate, momentum, L2_reg);
      stop_timer(12);

      // Input word embeddings

      start_timer(13);
      input_layer_node.param->computeGradient(first_hidden_linear_node.bProp_matrix,
                                              data,
                                              learning_rate, momentum, L2_reg);
      stop_timer(13);
    } else if (parameter_update == "ADA") {
      if (!skip_hidden)
      {
        start_timer(10);
        second_hidden_linear_node.param->computeGradientAdagrad(second_hidden_activation_node.bProp_matrix,
                                                                first_hidden_activation_node.fProp_matrix,
                                                                learning_rate,
                                                                L2_reg);
        stop_timer(10);
      }

      // First hidden layer


      start_timer(12);
      first_hidden_linear_node.param->computeGradientAdagrad(first_hidden_activation_node.bProp_matrix,
                                                             input_layer_node.fProp_matrix,
                                                             learning_rate,
                                                             L2_reg);
      stop_timer(12);

      // Input word embeddings

      start_timer(13);
      input_layer_node.param->computeGradientAdagrad(first_hidden_linear_node.bProp_matrix,
                                                     data,
                                                     learning_rate,
                                                     L2_reg);
      stop_timer(13);
    } else if (parameter_update == "ADAD") {
      int current_minibatch_size = first_hidden_activation_node.fProp_matrix.cols();
      //std::cerr<<"Adadelta gradient"<<endl;
      if (!skip_hidden)
      {
        start_timer(10);
        second_hidden_linear_node.param->computeGradientAdadelta(second_hidden_activation_node.bProp_matrix,
                                                                 first_hidden_activation_node.fProp_matrix,
                                                                 1.0/current_minibatch_size,
                                                                 L2_reg,
                                                                 conditioning_constant,
                                                                 decay);
        stop_timer(10);
      }
      //std::cerr<<"Finished gradient for second hidden linear layer"<<std::endl;

      // First hidden layer


      start_timer(12);
      first_hidden_linear_node.param->computeGradientAdadelta(first_hidden_activation_node.bProp_matrix,
                                                              input_layer_node.fProp_matrix,
                                                              1.0/current_minibatch_size,
                                                              L2_reg,
                                                              conditioning_constant,
                                                              decay);
      stop_timer(12);

      //std::cerr<<"Finished gradient for first hidden linear layer"<<std::endl;
      // Input word embeddings

      start_timer(13);
      input_layer_node.param->computeGradientAdadelta(first_hidden_linear_node.bProp_matrix,
                                                      data,
                                                      1.0/current_minibatch_size,
                                                      L2_reg,
                                                      conditioning_constant,
                                                      decay);
      stop_timer(13);

      //std::cerr<<"Finished gradient for first input layer"<<std::endl;
    } else {
      std::cerr<<"Parameter update :"<<parameter_update<<" is unrecognized"<<std::endl;
    }

  }
};

} // namespace nplm

#endif
