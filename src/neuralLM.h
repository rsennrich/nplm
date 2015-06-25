#ifndef NEURALLM_H
#define NEURALLM_H

#include <vector>
#include <cctype>
#include <cstdlib>
#include <boost/shared_ptr.hpp>

#include <Eigen/Dense>

#include "util.h"
#include "vocabulary.h"
#include "neuralNetwork.h"
#include "replace_digits.hpp"

/*
  To do:
  - move digit mapping into vocabulary.h
*/

namespace nplm
{

class neuralLM : public neuralNetwork, graehl::replace_digits
{
  boost::shared_ptr<vocabulary> vocab;
  int start, null;

 public:
  neuralLM()
      : neuralNetwork(),
        graehl::replace_digits(0),
        vocab(new vocabulary())
  {
  }

  void set_map_digits(char value) { map_digits = value; }

  void set_vocabulary(const vocabulary &vocab)
  {
    *(this->vocab) = vocab;
    start = vocab.lookup_word("<s>");
    null = vocab.lookup_word("<null>");
  }

  const vocabulary &get_vocabulary() const { return *(this->vocab); }


  int lookup_word(const std::string &word) const
  {
    if (map_digits)
      for (int i=0, n=word.size(); i<n; ++i)
        if (graehl::ascii_digit(word[i])) {
          std::string mapped_word(word);
          replace(mapped_word, i);
          return vocab->lookup_word(mapped_word);
        }
    return vocab->lookup_word(word);
  }

  int lookup_word(std::pair<char const*, char const*> slice) const
  {
    if (map_digits)
      for (char const* i = slice.first; i != slice.second; ++i)
        if (graehl::ascii_digit(*i)) {
          std::string mapped_word(slice.first, slice.second);
          replace(mapped_word, i - slice.first);
          return vocab->lookup_word(mapped_word);
        }
    return vocab->lookup_word(slice);
  }

  double lookup_ngram(const int *ngram_a, int n)
  {
    Eigen::Matrix<int,Eigen::Dynamic,1> ngram(m->ngram_size);
    for (int i=0; i<m->ngram_size; ++i)
    {
      if (i-m->ngram_size+n < 0)
      {
        if (ngram_a[0] == start)
          ngram(i) = start;
        else
          ngram(i) = null;
      }
      else
      {
        ngram(i) = ngram_a[i-m->ngram_size+n];
      }
    }
    return neuralNetwork::lookup_ngram(ngram);
  }

  double lookup_ngram(const std::vector<int> &ngram_v)
  {
    return lookup_ngram(ngram_v.data(), ngram_v.size());
  }

  template <typename Derived>
  double lookup_ngram(const Eigen::MatrixBase<Derived> &ngram)
  {
    return neuralNetwork::lookup_ngram(ngram);
  }

  template <typename DerivedA, typename DerivedB>
  void lookup_ngram(const Eigen::MatrixBase<DerivedA> &ngram, const Eigen::MatrixBase<DerivedB> &log_probs_const)
  {
    return neuralNetwork::lookup_ngram(ngram, log_probs_const);
  }

  void read(const std::string &filename)
  {
    std::vector<std::string> words;
    m->read(filename, words);
    set_vocabulary(vocabulary(words));
    resize();
    // this is faster but takes more memory
    //m->premultiply();
  }

};

template <typename T>
void addStartStop(std::vector<T> &input, std::vector<T> &output, int ngram_size, const T &start, const T &stop)
{
  output.clear();
  output.resize(input.size()+ngram_size);
  for (int i=0; i<ngram_size-1; ++i)
    output[i] = start;
  std::copy(input.begin(), input.end(), output.begin()+ngram_size-1);
  output[output.size()-1] = stop;
}

template <typename T>
void makeNgrams(const std::vector<T> &input, std::vector<std::vector<T> > &output, int ngram_size)
{
  output.clear();
  for (int j=ngram_size-1; j<input.size(); j++)
  {
    std::vector<T> ngram(input.begin() + (j-ngram_size+1), input.begin() + j+1);
    output.push_back(ngram);
  }
}

inline void preprocessWords(const std::vector<std::string> &words,
                            std::vector< std::vector<int> > &ngrams,
                            int ngram_size,
                            const vocabulary &vocab,
                            bool numberize,
                            bool add_start_stop,
                            bool ngramize) {
  int start = vocab.lookup_word("<s>");
  int stop = vocab.lookup_word("</s>");

  // convert words to ints
  std::vector<int> nums;
  if (numberize) {
    for (int j=0; j<words.size(); j++) {
      nums.push_back(vocab.lookup_word(words[j]));
    }
  }
  else {
    for (int j=0; j<words.size(); j++) {
      nums.push_back(boost::lexical_cast<int>(words[j]));
    }
  }

  // convert sequence to n-grams
  ngrams.clear();
  if (ngramize) {
    std::vector<int> snums;
    if (add_start_stop) {
      addStartStop<int>(nums, snums, ngram_size, start, stop);
    } else {
      snums = nums;
    }
    makeNgrams(snums, ngrams, ngram_size);
  }
  else {
    if (nums.size() != ngram_size)
    {
      std::cerr << "error: wrong number of fields in line\n";
      std::exit(1);
    }
    ngrams.push_back(nums);
  }
}

} // namespace nplm

#endif
