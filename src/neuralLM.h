#ifndef NEURALLM_H
#define NEURALLM_H

#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <cctype>
#include <cstdlib>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>

#include "../3rdparty/Eigen/Dense"

#include "param.h"
#include "util.h"
#include "model.h"
#include "propagator.h"
#include "neuralClasses.h"
#include "vocabulary.h"

#ifdef WITH_THREADS // included in multi-threaded moses project
#include <boost/thread/shared_mutex.hpp>
#endif

using namespace std;

namespace nplm {

class neuralLMShared {
#ifdef WITH_THREADS
  mutable boost::shared_mutex m_cacheLock;
#endif

 public:
  vocabulary input_vocab, output_vocab;
  model nn;

  size_t cache_size;
  Eigen::Matrix<int, Dynamic, Dynamic> cache_keys;
  vector<double> cache_values;
  int cache_lookups, cache_hits;

  explicit neuralLMShared(const string &filename, bool premultiply = false)
     : cache_size(0) {
    vector<string> input_words, output_words;
    nn.read(filename, input_words, output_words);
    input_vocab = vocabulary(input_words);
    output_vocab = vocabulary(output_words);
    // this is faster but takes more memory
    if (premultiply) {
      nn.premultiply();
    }
    if (cache_size) {
      cache_keys.resize(nn.ngram_size, cache_size);
      cache_keys.fill(-1);
    }
  }

  template <typename Derived>
  double lookup_cache(const Eigen::MatrixBase<Derived> &ngram) {
    size_t hash;
    if (cache_size) {
      // First look in cache
      hash = Eigen::hash_value(ngram) % cache_size; // defined in util.h
      cache_lookups++;
#ifdef WITH_THREADS // wait until nobody writes to cache
      boost::shared_lock<boost::shared_mutex> read_lock(m_cacheLock);
#endif
      if (cache_keys.col(hash) == ngram) {
        cache_hits++;
        return cache_values[hash];
      }
    }

    return 0;
  }

  template <typename Derived>
  void store_cache(const Eigen::MatrixBase<Derived> &ngram, double log_prob) {
    size_t hash;
    if (cache_size) {
      hash = Eigen::hash_value(ngram) % cache_size;
#ifdef WITH_THREADS // block others from reading cache
      boost::unique_lock<boost::shared_mutex> lock(m_cacheLock);
#endif
      // Update cache
      cache_keys.col(hash) = ngram;
      cache_values[hash] = log_prob;
    }
  }

  void set_cache(size_t cache_size) {
    this->cache_size = cache_size;
    cache_keys.resize(nn.ngram_size, cache_size);
    cache_keys.fill(-1); // clears cache
    cache_values.resize(cache_size);
    cache_lookups = cache_hits = 0;
  }
};

class neuralLM {
  // Big stuff shared across instances.
  boost::shared_ptr<neuralLMShared> shared;

  bool normalization;
  char map_digits;

  propagator prop;

  int ngram_size;
  int width;

  double weight;

  Eigen::Matrix<int,Eigen::Dynamic,1> ngram; // buffer for lookup_ngram
  int start, null;

 public:
  neuralLM(const string &filename, bool premultiply = false)
      : shared(new neuralLMShared(filename, premultiply)),
        ngram_size(shared->nn.ngram_size),
        normalization(false),
        weight(1.),
        map_digits(0),
        width(1),
        prop(shared->nn, 1),
        start(shared->input_vocab.lookup_word("<s>")),
        null(shared->input_vocab.lookup_word("<null>")) {
    ngram.setZero(ngram_size);
    prop.resize();
  }

  // initialize neuralLM class that shares vocab and model with base instance (for multithreaded decoding)
  neuralLM(const neuralLM &baseInstance)
    : shared(baseInstance.shared),
      ngram_size(shared->nn.ngram_size),
      normalization(false),
      weight(1.),
      map_digits(0),
      width(1),
      prop(shared->nn, 1),
      start(shared->input_vocab.lookup_word("<s>")),
      null(shared->input_vocab.lookup_word("<null>")) {
    ngram.setZero(ngram_size);
    prop.resize();
  }

  void set_normalization(bool value) {
    normalization = value;
  }

  void set_log_base(double value) {
    weight = 1./log(value);
  }

  void set_map_digits(char value) {
    map_digits = value;
  }

  void set_width(int width) {
    this->width = width;
    prop.resize(width);
  }

  const vocabulary& get_vocabulary() const {
    return shared->input_vocab;
  }

  int lookup_input_word(const string &word) const {
    if (!map_digits) {
      return shared->input_vocab.lookup_word(word);
    }

    string mapped_word(word);
    for (int i = 0; i < word.length(); i++) {
      if (isdigit(word[i])) {
        mapped_word[i] = map_digits;
      }
    }

    return shared->input_vocab.lookup_word(mapped_word);
  }

  int lookup_word(const string &word) const {
    return lookup_input_word(word);
  }

  int lookup_output_word(const string &word) const {
    if (!map_digits) {
      return shared->output_vocab.lookup_word(word);
    }

    string mapped_word(word);
    for (int i = 0; i < word.length(); ++i) {
      if (isdigit(word[i])) {
        mapped_word[i] = map_digits;
      }
    }

    return shared->output_vocab.lookup_word(mapped_word);
  }

  Eigen::Matrix<int,Eigen::Dynamic,1> &staging_ngram() {
    return ngram;
  }

  double lookup_from_staging() {
    return lookup_ngram(ngram);
  }

  template <typename Derived>
  double lookup_ngram(const Eigen::MatrixBase<Derived> &ngram) {
    assert (ngram.rows() == ngram_size);
    assert (ngram.cols() == 1);

    double cached = shared->lookup_cache(ngram);
    if (cached != 0) {
      return cached;
    }

    // Make sure that we're single threaded. Multithreading doesn't help,
    // and in some cases can hurt quite a lot
    int save_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    int save_eigen_threads = Eigen::nbThreads();
    Eigen::setNbThreads(1);
#ifdef __INTEL_MKL__
    int save_mkl_threads = mkl_get_max_threads();
    mkl_set_num_threads(1);
#endif

    prop.fProp(ngram.col(0));

    int output = ngram(ngram_size-1, 0);
    double log_prob;

    start_timer(3);
    if (normalization) {
      Eigen::Matrix<double,Eigen::Dynamic,1> scores(shared->output_vocab.size());
      prop.output_layer_node.param->fProp(prop.second_hidden_activation_node.fProp_matrix, scores);
      double logz = logsum(scores.col(0));
      log_prob = weight * (scores(output, 0) - logz);
    } else {
      log_prob = weight * prop.output_layer_node.param->fProp(
          prop.second_hidden_activation_node.fProp_matrix, output, 0);
    }
    stop_timer(3);

    shared->store_cache(ngram, log_prob);

#ifndef WITH_THREADS
#ifdef __INTEL_MKL__
    mkl_set_num_threads(save_mkl_threads);
#endif
    Eigen::setNbThreads(save_eigen_threads);
    omp_set_num_threads(save_threads);
#endif
    return log_prob;
  }

  // Look up many n-grams in parallel.
  template <typename DerivedA, typename DerivedB>
  void lookup_ngram(
      const Eigen::MatrixBase<DerivedA> &ngram,
      const Eigen::MatrixBase<DerivedB> &log_probs_const) {
    UNCONST(DerivedB, log_probs_const, log_probs);
    assert(ngram.rows() == ngram_size);
    assert(ngram.cols() <= width);

    prop.fProp(ngram);

    if (normalization) {
      Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> scores(
          shared->output_vocab.size(), ngram.cols());
      prop.output_layer_node.param->fProp(
          prop.second_hidden_activation_node.fProp_matrix, scores);

      // And softmax and loss
      Matrix<double,Dynamic,Dynamic> output_probs(
          shared->nn.output_vocab_size, ngram.cols());
      double minibatch_log_likelihood;
      SoftmaxLogLoss().fProp(
          scores.leftCols(ngram.cols()), ngram.row(shared->nn.ngram_size-1),
          output_probs, minibatch_log_likelihood);

      for (int j = 0; j < ngram.cols(); j++) {
        int output = ngram(ngram_size - 1, j);
        log_probs(0, j) = weight * output_probs(output, j);
      }
    } else {
      for (int j = 0; j < ngram.cols(); j++) {
        int output = ngram(ngram_size - 1, j);
        log_probs(0, j) = weight * prop.output_layer_node.param->fProp(
            prop.second_hidden_activation_node.fProp_matrix, output, j);
      }
    }
  }

  double lookup_ngram(const int *ngram_a, int n) {
    for (int i = 0; i < ngram_size; i++) {
      if (i - ngram_size + n < 0) {
        if (ngram_a[0] == start) {
          ngram(i) = start;
        } else {
          ngram(i) = null;
        }
      } else {
        ngram(i) = ngram_a[i - ngram_size + n];
      }
    }
    return lookup_ngram(ngram);
  }

  double lookup_ngram(const vector<int> &ngram_v) {
    return lookup_ngram(ngram_v.data(), ngram_v.size());
  }

  int get_order() const {
    return ngram_size;
  }

   void set_cache(size_t cache_size) {
    shared->set_cache(cache_size);
  }

  double cache_hit_rate() {
    return static_cast<double>(shared->cache_hits) / shared->cache_lookups;
  }
};

} // namespace nplm

#endif
