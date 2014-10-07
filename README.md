### Getting started

Before compiling, you must have the following:

* [g++](https://gcc.gnu.org/onlinedocs/gcc-3.3.6/gcc/G_002b_002b-and-GCC.html) 4.6.3 or later
* [GNU make](http://www.gnu.org/software/make/)
* [Boost](http://www.boost.org) 1.47.0 or later

Decoders (only needed if you want to use NPLM in that particular SMT system):

* [Moses](http://www.statmt.org/moses/)

Optional:

* [Intel MKL](http://software.intel.com/en-us/intel-mkl) 11.x (recommended for better performance)
* [Python](http://python.org) 2.7.x, not 3.x
* [Cython](http://cython.org) 0.19.x (needed only for building python bindings)

### Building

To compile, edit the Makefile to reflect the locations of the Boost include directories.

By default, multithreading using OpenMP is enabled. To turn it off,
comment out the line `OMP=1`.

Run the following commands to compile the code:

    cd src
    make

This creates several executables and the `neuralLM..a` library in the `src` directory.

### Prepare the training data

First tokenize, lowercase, etc. your training data using your favourite tools. After that, run:

    nplm/src/prepareNeuralLM --train_text training.en \
                             --train_file training.ngrams \
                             --validation_text dev.en \
                             --validation_file dev.ngrams \
                             --ngram_size 5 \
                             --vocab_size <vocab_size> \
                             --write-words-file training.vocab

Set `vocab-size` to a value lower than the true vocabulary size, so the model can learn a representation for unknown words (marked with the `<unk>` token).

This script does the following:
- It creates a vocabulary of the `vocab-size` most frequent words, mapping all other
  words to `<unk>`.
- Adds start `<s>` and stop `</s>` symbols.
- Converts both the training data and the validation data to numberized n-grams (one per line).

### Train a language model

Run:

    trainNeuralNetwork --train_file train.ngrams \
                       --validation_file dev.ngrams \
                       --num_epochs 10 \
                       --words_file training.vocab \
                       --model_prefix nplm

After each pass through the data, the trainer will print the
log-likelihood of both the training data and validation data (higher
is better) and generate a series of model files called model.1,
model.2, and so on. You choose which model you want based on the
validation log-likelihood.

Notes:

- Normalization. Most of the computational cost normally (no pun
 intended) associated with a large vocabulary has to do with
  normalization of the conditional probability distribution `P(word |
  context)`. The trainer uses noise-contrastive estimation to avoid
  this cost during training [(Gutmann and Hyvärinen, 2010)](http://jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf), and, by
  default, sets the normalization factors to one to avoid this cost
  during testing [(Mnih and Teh, 2012)](http://www.cs.toronto.edu/~amnih/papers/ncelm.pdf).

  If you set `--normalization 1`, the trainer will try to learn the
  normalization factors, and you should accordingly turn on
  normalization when using the resulting model. The default initial
  value `--normalization_init 0` should be fine; you can try setting it
  a little higher, but not lower.

- Validation. The trainer computes the log-likelihood of a validation
  data set (which should be disjoint from the training data). You use
  this to decide when to stop training, and the trainer also uses it
  to throttle the learning rate. This computation always uses exact
  normalization and is therefore much slower, per instance, than
  training. Therefore, you should not use a large test corpus for
  validation (e.g. ~50k tokens).

### Decoding with Moses

A [feature function](https://github.com/moses-smt/mosesdecoder/tree/master/moses/LM) for the Moses decoder has already been pushed in the [Moses
repository](https://github.com/moses-smt/mosesdecoder). To use it, you must first compile Moses as follows:

    ./bjam -j <num_threads> --with-nplm=</absolute/path/to/nplm>

### Miscellaneous

This section mainly contains information about features that I haven't tested since I forked the project.

#### MKL

If you want to use the Intel MKL library (recommended if you have it), set `MKL` to point to the MKL root directory.

#### Troubleshooting

- Intel C++ compiler and OpenMP. With version 12, you may get a
  "pragma not found" error. This is reportedly fixed in ComposerXE
  update 9.

- Mac OS X and OpenMP. The Clang compiler (`/usr/bin/c++`) doesn't
  support OpenMP. If the g++ that comes with XCode doesn't work
  either, try the one installed by MacPorts (`/opt/local/bin/g++` or
  `/opt/local/bin/g++-mp-*`).

#### Python code

prepareNeuralLM.py performs the same function as prepareNeuralLM, but in
Python. This may be handy if you want to make modifications.

nplm.py is a pure Python module for reading and using language models
created by trainNeuralNetwork. See testNeuralLM.py for example usage.

In src/python are Python bindings (using Cython) for the C++ code. To
build them, run `make python/nplm.so`.

#### Writing your own deocder feature function

To use the language model in a decoder, include neuralLM.h and link
against neuralLM.a. This provides a class nplm::neuralLM, with the
following methods:

    void set_normalization(bool normalization);

Turn normalization on or off (default: off). If normalization is off,
the probabilities output by the model will not be normalized. In
general, this means that summing over all possible words will not give
a probability of one. If normalization is on, computes exact
probabilities (too slow to be recommended for decoding).

    void set_map_digits(char c);

Map all digits (0-9) to the specified character. This should match
whatever mapping you used during preprocessing.

    void set_log_base(double base);

Set the base of the log-probabilities returned by `lookup_ngram`. The
default is e (natural log), whereas most other language modeling
toolkits use base 10.

    void read(const string &filename);

Read model from file.

    int get_order();

Return the order of the language model.

    int lookup_word(const string &word);

Map a word to an index for use with `lookup_ngram()`.

    double lookup_ngram(const vector<int> &ngram);
    double lookup_ngram(const int *ngram, int n);

Look up the log-probability of ngram.
