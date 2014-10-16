#include <iostream>
#include <vector>

#include "vocabulary.h"

using namespace std;

namespace nplm {

// Adds ngram_size - 1 "<s>" symbols at the beginning of the sentence and a
// "</s>" symbol at the end of the sentence. This is because we want to create
// n-grams starting with the first word in the sentence.
template <typename T>
void addStartStop(
    std::vector<T> &input,
    std::vector<T> &output,
    int ngram_size,
    const T &start,
    const T &stop) {
  output.clear();
  output.resize(input.size() + ngram_size);
  for (int i = 0; i < ngram_size - 1; i++) {
    output[i] = start;
  }

  std::copy(input.begin(), input.end(), output.begin() + ngram_size - 1);
  output[output.size() - 1] = stop;
}

// Generates all the n-grams in a sentence (after the sentence has been
// properly padded with <s> and </s>).
template <typename T>
void makeNgrams(
    const std::vector<T> &input,
    std::vector<std::vector<T> > &output,
    int ngram_size) {
  output.clear();
  for (int j = ngram_size - 1; j < input.size(); j++) {
    std::vector<T> ngram(
        input.begin() + j - ngram_size + 1, input.begin() + j + 1);
    output.push_back(ngram);
  }
}

inline void preprocessWords(
    const std::vector<std::string> &words,
    std::vector<std::vector<int> > &ngrams,
    int ngram_size, const vocabulary &vocab,
    bool numberize, bool add_start_stop, bool ngramize) {
  int start = vocab.lookup_word("<s>");
  int stop = vocab.lookup_word("</s>");

  // convert words to ints
  std::vector<int> nums;
  if (numberize) {
    for (int j = 0; j < words.size(); j++) {
      nums.push_back(vocab.lookup_word(words[j]));
    }
  } else {
    for (int j = 0; j < words.size(); j++) {
      nums.push_back(boost::lexical_cast<int>(words[j]));
    }
  }

  ngrams.clear();
  if (ngramize) {
    // Convert sentence to n-grams.
    std::vector<int> snums;
    if (add_start_stop) {
      addStartStop<int>(nums, snums, ngram_size, start, stop);
    } else {
      snums = nums;
    }
    makeNgrams(snums, ngrams, ngram_size);
  } else {
    // The line contains only a single n-gram.
    if (nums.size() != ngram_size) {
      std::cerr << "error: wrong number of fields in line" << std::endl;
      std::exit(1);
    }
    ngrams.push_back(nums);
  }
}

void writeNgrams(
    const vector<vector<string> >& data, int ngram_size,
    const vocabulary &vocab, bool numberize, bool add_start_stop,
    bool ngramize, const string &filename) {
  ofstream file(filename.c_str());
  if (!file) {
    cerr << "error: could not open " << filename << endl;
    exit(1);
  }

  vector<vector<int> > ngrams;
  for (int i = 0; i < data.size(); i++) {
    preprocessWords(data[i], ngrams, ngram_size, vocab, numberize, add_start_stop, ngramize);

    // write out n-grams
    for (int j = 0; j < ngrams.size(); j++) {
      for (int k = 0; k < ngram_size; k++) {
        file << ngrams[j][k] << " ";
      }
      file << endl;
    }
  }
  file.close();
}

} // namespace nplm
