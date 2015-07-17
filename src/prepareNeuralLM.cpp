#include <iostream>
#include <vector>
#include <queue>
#include <deque>
#include <fstream>
#include <iterator>

#include <boost/unordered_map.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <tclap/CmdLine.h>

#include "neuralLM.h"
#include "util.h"

using namespace std;
using namespace TCLAP;
using namespace boost;
using namespace nplm;
using namespace boost::random;
namespace ip = boost::interprocess;

typedef ip::allocator<int, ip::managed_mapped_file::segment_manager> intAllocator;
typedef ip::vector<int, intAllocator> vec;
typedef ip::allocator<vec, ip::managed_mapped_file::segment_manager> vecAllocator;
//typedef allocator<ValueType, managed_shared_memory::segment_manager> ShmemAllocator;
//typedef multimap<int, vec, std::less<int>, ShmemAllocator> MyMap;
typedef std::vector<vec,vecAllocator> vecvec;

typedef long long int data_size_t; // training data can easily exceed 2G instances

template<typename T>
void writeNgrams(const T &data,
                 int ngram_size,
                 const vocabulary &vocab,
                 bool numberize,
                 bool add_start_stop,
                 bool ngramize,
                 const string &filename)
{
  ofstream file(filename.c_str());
  if (!file)
  {
    cerr << "error: could not open " << filename << endl;
    exit(1);
  }

  vector<vector<int> > ngrams;

  for (int i=0; i<data.size(); i++) {
    preprocessWords(data[i], ngrams, ngram_size, vocab, numberize, add_start_stop, ngramize);
    // write out n-grams
    for (int j=0; j<ngrams.size(); j++)
    {
      for (int k=0; k<ngram_size; k++)
      {
        file << ngrams[j][k] << " ";
      }
      file << endl;
    }
  }
  file.close();
}

// Space efficient version for writing the n-grams.
// They are not read into memory.
void writeNgrams(const string &input_filename,
                 int ngram_size,
                 const vocabulary &vocab,
                 bool numberize,
                 bool add_start_stop,
                 bool ngramize,
                 const string &filename,
                 int train_data_size,
                 vector<float> &sent_weights,
                 const string &sent_weights_filename)
{
  ofstream file(filename.c_str());
  ofstream output_sent_weights_file(sent_weights_filename.c_str());
  if (!file)
  {
    cerr << "error: could not open " << filename << endl;
    exit(1);
  }

  ifstream input_file(input_filename.c_str());
  vector<vector<int> > ngrams;
  //for (int i=0; i<train_data.size(); i++) {
  string line;
  int counter = 0;
  cerr<<"Processed ... ";
  while (getline(input_file,line) && train_data_size-- > 0) {
    counter++;
    if ((counter % 100000) == 0) {
      cerr<<counter<<" training lines ... ";
    }
    //stringstream lstr(line);
    vector<string> lstr_items;
    splitBySpace(line,lstr_items);

    //for (int i=0; i<data.size(); i++) {
    preprocessWords(lstr_items,
                    ngrams,
                    ngram_size,
                    vocab,
                    numberize,
                    add_start_stop,
                    ngramize);

    // write out n-grams
    for (int j=0; j<ngrams.size(); j++)
    {
      if (sent_weights.size() != 0) {
        output_sent_weights_file <<sent_weights[counter-1]<<endl;
      }
      for (int k=0; k<ngram_size; k++)
      {
        file << ngrams[j][k] << " ";
      }
      file << endl;
    }
  }
  cerr<<endl;
  input_file.close();
  file.close();
  output_sent_weights_file.close();
}

// Space efficient version for writing the n-grams.
// They are not read into memory.
void writeMmapNgrams(const string &input_filename,
                     int ngram_size,
                     const vocabulary &vocab,
                     bool numberize,
                     bool add_start_stop,
                     bool ngramize,
                     const string &filename,
                     unsigned long train_data_size,
                     data_size_t num_tokens,
                     bool randomize)
{
  cerr<<"Num tokens is "<<num_tokens<<endl;
  cerr<<"Training data size is "<<train_data_size<<endl;
  // Open the memory mapped file and create the allocators
  ip::managed_mapped_file mfile(ip::create_only,
                                filename.c_str(),
                                num_tokens*ngram_size*sizeof(int)+1024UL*1024UL);
  intAllocator ialloc(mfile.get_segment_manager());
  vecAllocator valloc (mfile.get_segment_manager());
  //vecvec *mMapVecVec= mfile.construct<vecvec>("data")(num_tokens,vec(ialloc),valloc);

  vec *mMapVec= mfile.construct<vec>("vector")(num_tokens*ngram_size,0,ialloc);

  cerr<<"The size of mmaped vec is "<<mMapVec->size()<<endl;
  // Going over every line in the input file and
  // printing the memory mapped ngrams into the
  // output file
  ifstream input_file(input_filename.c_str());
  //for (int i=0; i<train_data.size(); i++) {
  string line;
  int counter = 0;
  cerr<<"Processed ... ";
  long int train_ngram_counter = 0;
  vector<vector<int> > ngrams;
  while (getline(input_file,line) && train_data_size-- > 0) {
    counter++;
    if ((counter % 100000) ==0) {
      //cerr<<"counter is "<<counter<<endl;
      cerr<<counter<<" training lines ... ";
    }
    //stringstream lstr(line);
    vector<string> lstr_items;
    splitBySpace(line,lstr_items);

    //for (int i=0; i<data.size(); i++) {
    preprocessWords(lstr_items, ngrams,
                    ngram_size,
                    vocab,
                    numberize,
                    add_start_stop,
                    ngramize);
    /*
      cerr<<"line is "<<endl;
      cerr<<line<<endl;
      cerr<<"Number of ngrams is "<<ngrams.size()<<endl;
      if (ngrams.size() ==1 ){
      cerr<<"The line number was "<<counter<<endl;
      cerr<<line<<endl;
      }
    */
    // write out n-grams in mmapped file
    for (int j=0; j<ngrams.size(); j++)
    {
      /*
        for (int k=0; k<ngram_size; k++)
        {
        cerr << ngrams[j][k] << " ";
        }
        cerr<< endl;
      */
      for (int k=0; k<ngram_size; k++) {
        mMapVec->at(train_ngram_counter*ngram_size+k) = ngrams[j][k];
      }
      train_ngram_counter++;
      //cerr<<"Train ngram counter is "<<train_ngram_counter<<endl;
    }
  }
  cerr<<endl;
  input_file.close();

  // Shrink the file if it was overused
  ip::managed_mapped_file::shrink_to_fit(filename.c_str());
  //now to randomize the items if the randomize flag was set
  if (randomize == true) {
    unsigned seed = 1234; //for testing only
    boost::random::mt19937 rng(seed);
    cerr<<"Randomly shuffling data...";
    data_size_t counter =0;
    while (counter < num_tokens) {
      data_size_t upper_limit = counter+5000000;
      long int vector_size = 5000000;
      if (counter + 10000000 >= num_tokens) {
        upper_limit = num_tokens;
        vector_size = num_tokens - counter;
      }
      vector<int> temp(vector_size*ngram_size,0);
      for (int i=0;i<vector_size;i++){
        for (int k=0;k<ngram_size;k++) {
          temp[i*ngram_size+k] = mMapVec->at((i+counter)*ngram_size+k);
        }
      }
      for (data_size_t i=vector_size-1; i>0; i--)
      {
        if (i %500000 == 0) {
          cerr<<"Shuffled "<<num_tokens-1<<" instances...";
        }
        data_size_t j = boost::random::uniform_int_distribution<data_size_t>(0, i-1)(rng);
        for (int k=0;k<ngram_size;k++) {
          int temp_val = temp.at(i*ngram_size+k);
          temp.at(i*ngram_size+k) =
              temp.at(j*ngram_size+k);
          temp.at(j*ngram_size+k) = temp_val;
        }
      }
      //Putting it back
      for (int i=0;i<vector_size;i++){
        for (int k=0;k<ngram_size;k++) {
          mMapVec->at((i+counter)*ngram_size+k) = temp[i*ngram_size+k];
        }
      }
      counter = upper_limit;
    }

    /*
      for (data_size_t i=num_tokens-1; i>0; i--)
      {
      if (i %500000 == 0) {
      cerr<<"Shuffled "<<num_tokens-1<<" instances...";
      }
      data_size_t j = boost::random::uniform_int_distribution<data_size_t>(0, i-1)(rng);
      for (int k=0;k<ngram_size;k++) {
      int temp_val = mMapVec->at(i*ngram_size+k);
      mMapVec->at(i*ngram_size+k) =
      mMapVec->at(j*ngram_size+k);
      mMapVec->at(j*ngram_size+k) = temp_val;
      }
      }
    */
    cerr<<endl;
  }
}


int main(int argc, char *argv[])
{
  ios::sync_with_stdio(false);
  int ngram_size, vocab_size, validation_size;
  bool numberize,
      ngramize,
      add_start_stop,
      mmap_file,
      randomize;

  string train_text,
      train_file,
      validation_text,
      validation_file,
      words_file,
      write_words_file,
      sent_weights_text,
      output_sent_weights_text;

  try
  {
    CmdLine cmd("Prepares training data for training a language model.", ' ', "0.1");

    // The options are printed in reverse order

    ValueArg<bool> arg_ngramize("", "ngramize", "If true, convert lines to ngrams. Default: true.", false, true, "bool", cmd);
    ValueArg<bool> arg_numberize("", "numberize", "If true, convert words to numbers. Default: true.", false, true, "bool", cmd);
    ValueArg<bool> arg_add_start_stop("", "add_start_stop", "If true, prepend <s> and append </s>. Default: true.", false, true, "bool", cmd);
    ValueArg<bool> arg_mmap_file("", "mmap_file", "If true, the training file will be a memory mapped file. \n This is "
                                 "needed if the entire training data cannot fit in memory. Default: false.", false, false, "bool", cmd);

    ValueArg<bool> arg_randomize("", "randomize", "If true, Randomly shuffle the training ngrams. It can only be used with mmap_file =1 . Default: false.", false, false, "bool", cmd);

    ValueArg<int> arg_vocab_size("", "vocab_size", "Vocabulary size.", false, -1, "int", cmd);
    ValueArg<string> arg_words_file("", "words_file", "File specifying words that should be included in vocabulary; all other words will be replaced by <unk>.", false, "", "string", cmd);
    ValueArg<int> arg_ngram_size("", "ngram_size", "Size of n-grams.", true, -1, "int", cmd);
    ValueArg<string> arg_write_words_file("", "write_words_file", "Output vocabulary.", false, "", "string", cmd);
    ValueArg<int> arg_validation_size("", "validation_size", "How many lines from training data to hold out for validation. Default: 0.", false, 0, "int", cmd);
    ValueArg<string> arg_validation_file("", "validation_file", "Output validation data (numberized n-grams).", false, "", "string", cmd);
    ValueArg<string> arg_validation_text("", "validation_text", "Input validation data (tokenized). Overrides --validation_size. Default: none.", false, "", "string", cmd);
    ValueArg<string> arg_train_file("", "train_file", "Output training data (numberized n-grams).", false, "", "string", cmd);
    ValueArg<string> arg_train_text("", "train_text", "Input training data (tokenized).", true, "", "string", cmd);
    //ValueArg<string> arg_sent_weights_text("", "sent_weights_text", "The sentence weights text", false, "", "string", cmd);
    //ValueArg<string> arg_sent_weights_file("", "sent_weights_file", "The file to write the per ngram weights", false, "", "string", cmd);


    cmd.parse(argc, argv);

    train_text = arg_train_text.getValue();
    train_file = arg_train_file.getValue();
    validation_text = arg_validation_text.getValue();
    validation_file = arg_validation_file.getValue();
    validation_size = arg_validation_size.getValue();
    write_words_file = arg_write_words_file.getValue();
    ngram_size = arg_ngram_size.getValue();
    vocab_size = arg_vocab_size.getValue();
    words_file = arg_words_file.getValue();
    numberize = arg_numberize.getValue();
    ngramize = arg_ngramize.getValue();
    add_start_stop = arg_add_start_stop.getValue();
    mmap_file = arg_mmap_file.getValue();
    randomize = arg_randomize.getValue();
    //sent_weights_text = arg_sent_weights_text.getValue();
    //output_sent_weights_text = arg_sent_weights_file.getValue();
    sent_weights_text = "";
    output_sent_weights_text = "";


    // check command line arguments

    // Notes:
    // - either --words_file or --vocab_size is required.
    // - if --words_file is set,
    // - if --vocab_size is not set, it is inferred from the length of the file
    // - if --vocab_size is set, it is an error if the vocab file has a different number of lines
    // - if --numberize 0 is set and --words_file f is not set, then the output model file will not have a vocabulary, and a warning should be printed.

    // Notes:
    // - if --ngramize 0 is set, then
    // - if --ngram_size is not set, it is inferred from the training file (different from current)
    // - if --ngram_size is set, it is an error if the training file has a different n-gram size
    // - if neither --validation_file or --validation_size is set, validation will not be performed.
    // - if --numberize 0 is set, then --validation_size cannot be used.

    cerr << "Command line: " << endl;
    cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << endl;

    const string sep(" Value: ");
    cerr << arg_train_text.getDescription() << sep << arg_train_text.getValue() << endl;
    cerr << arg_train_file.getDescription() << sep << arg_train_file.getValue() << endl;
    cerr << arg_validation_text.getDescription() << sep << arg_validation_text.getValue() << endl;
    cerr << arg_validation_file.getDescription() << sep << arg_validation_file.getValue() << endl;
    cerr << arg_validation_size.getDescription() << sep << arg_validation_size.getValue() << endl;
    cerr << arg_write_words_file.getDescription() << sep << arg_write_words_file.getValue() << endl;
    cerr << arg_ngram_size.getDescription() << sep << arg_ngram_size.getValue() << endl;
    cerr << arg_vocab_size.getDescription() << sep << arg_vocab_size.getValue() << endl;
    cerr << arg_words_file.getDescription() << sep << arg_words_file.getValue() << endl;
    cerr << arg_numberize.getDescription() << sep << arg_numberize.getValue() << endl;
    cerr << arg_ngramize.getDescription() << sep << arg_ngramize.getValue() << endl;
    cerr << arg_add_start_stop.getDescription() << sep << arg_add_start_stop.getValue() << endl;
    cerr << arg_mmap_file.getDescription() << sep << arg_mmap_file.getValue() << endl;
    //cerr << arg_sent_weights_text.getDescription() << sep << arg_sent_weights_text.getValue() << endl;
  }
  catch (TCLAP::ArgException &e)
  {
    cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
    exit(1);
  }

  // VLF: why is this true?
  // DC: it's because the vocabulary has to be constructed from the training data only.
  // If the vocabulary is preset, we can't create the validation data.
  // - if --numberize 0 is set, then --validation_size cannot be used.
  // if (!numberize && (validation_size > 0)) {
  //     cerr <<  "Warning: without setting --numberize to 1, --validation_size cannot be used." << endl;
  // }

  // Read in training data and validation data
  // vector<vector<string> > train_data;
  // readSentFile(train_text, train_data);
  // @vaswani: No more reading the entire training file into memory
  // Reading it per line with file io

  //for (int i=0; i<train_data.size(); i++) {
  // Go over every line in the file and
  // 1. if the !ngramize then you should check if
  // we have the correct number of items per line
  // 2. build the vocabulary if the words file has not
  // been specified.
  // Construct vocabulary
  vocabulary vocab;
  int start, stop;
  // Add start stop if the vocabulary has not been supplied
  if (words_file == "") {
    vocab.insert_word("<s>");
    vocab.insert_word("</s>");
    vocab.insert_word("<null>");
    // warn user that if --numberize is not set, there will be no vocabulary!
    if (!numberize) {
      cerr << "Warning: with --numberize 0 and --words_file == "", there will be no vocabulary!" << endl;
    }
  }
  if (mmap_file == false && randomize == true) {
    cerr<<"Randomize option can only be used with mmap_file = 1"<<endl;
    exit(1);
  }
  unordered_map<string,int> count; // For keeping word counts if no supplied vocab

  deque<vector<string> > validation_data;
  int train_data_size=0;
  cerr<<"Processed ... ";
  data_size_t num_tokens=0;

  ifstream training(train_text.c_str());

  string line;
  while (getline(training,line)) {
    train_data_size++;
    //stringstream lstr(line);
    vector<string> lstr_items;
    splitBySpace(line,lstr_items);
    // if data is already ngramized, set/check ngram_size
    if (!ngramize) {
      if (ngram_size > 0) {
        if (ngram_size != lstr_items.size()) {
          cerr << "Error: size of training ngrams does not match specified value of --ngram_size!" << endl;
        }
      }
      // else if --ngram_size has not been specified, set it now
      else {
        ngram_size=lstr_items.size();
      }
    }
    if ((train_data_size%100000)==0){
      cerr<<train_data_size<<" lines ... ";
    }
    //break;
    /*
      if (lstr_items.size() ==1) {
      cerr<<"line :"<<endl;
      cerr<<line<<endl;
      cerr<<"The number of items was 1"<<endl;
      getchar();
      }
    */
    num_tokens += lstr_items.size()+1;
    if (words_file == "") {
      for (int j=0; j<lstr_items.size(); j++) {
        count[lstr_items[j]] += 1;
      }
    }
    // Add to validation set if the validation size
    // has not been specified
    if (validation_text == "" && validation_size > 0) {
      //cerr<<"validation size is "<<validation_data.size()<<endl;
      if (validation_data.size() == validation_size) {
        //validation_data.erase(validation_data.begin());
        validation_data.pop_front();
      }
      validation_data.push_back(lstr_items);
    }
  }
  cerr<<endl;
  training.close();
  //cerr<<"validation size is "<<validation_data.size()<<endl;
  //getchar();
  if (validation_data.size() < validation_size) {
    cerr<<"validation size is "<<validation_data.size()<<endl;
    cerr << "error: requested validation size is greater than training data size" << endl;
    exit(1);
  }

  train_data_size -= validation_size;
  cerr<<"Training data size is "<<train_data_size<<endl;

  // The items in the validation data have already been counted
  // Decrementing the counts of those words before building the vocabulary
  for(int i=0; i<validation_data.size(); i++){
    num_tokens -= (validation_data[i].size() +1);
    for (int j=0; j<validation_data[i].size();j++){
      count[validation_data[i][j]] -= 1;
      if (count[validation_data[i][j]] == 0) {
        count.erase(validation_data[i][j]);
      }
    }
  }

  // Getting the top n frequent words for the vocabulary
  if (words_file == "") {
    vocab.insert_most_frequent(count, vocab_size);
    if (vocab.size() < vocab_size) {
      cerr << "warning: fewer than " << vocab_size << " types in training data; the unknown word will not be learned" << endl;
    }
  }
  //vector<vector<string> > validation_data;
  if (validation_text != "") {
    readSentFile(validation_text, validation_data);
    for (int i=0; i<validation_data.size(); i++) {
      // if data is already ngramized, set/check ngram_size
      if (!ngramize) {
        // if --ngram_size has been specified, check that it does not conflict with --ngram_size
        if (ngram_size > 0) {
          if (ngram_size != validation_data[i].size()) {
            cerr << "Error: size of validation ngrams does not match specified value of --ngram_size!" << endl;
          }
        }
        // else if --ngram_size has not been specified, set it now
        else {
          ngram_size=validation_data[i].size();
        }
      }
    }
  }
  //READING SENTENCE WEIGHTS IF THERE ARE ANY
  vector<float> sent_weights;
  if (sent_weights_text != "") {
    cerr<<"Reading sentence weights from "<<sent_weights_text<<endl;
    ifstream sent_weights_file(sent_weights_text.c_str());
    string line;
    readWeightsFile(sent_weights_file,sent_weights);
    sent_weights_file.close();
    if (sent_weights_text.size() != train_data_size) {
      cerr<<"The number of sentence weights does not match the number of training sentences"<<endl;
    }
  }

  /*
    else if (validation_size > 0)
    {
    // Create validation data
    if (validation_size > train_data.size())
    {
    cerr << "error: requested validation size is greater than training data size" << endl;
    exit(1);
    }
    validation_data.insert(validation_data.end(), train_data.end()-validation_size, train_data.end());
    train_data.resize(train_data.size() - validation_size);
    }
  */

  // Construct vocabulary
  //vocabulary vocab;
  //int start, stop;

  // read vocabulary from file
  if (words_file != "") {
    vector<string> words;
    readWordsFile(words_file,words);
    for(vector<string>::iterator it = words.begin(); it != words.end(); ++it) {
      vocab.insert_word(*it);
    }

    // was vocab_size set? if so, verify that it does not conflict with size of vocabulary read from file
    if (vocab_size > 0) {
      if (vocab.size() != vocab_size) {
        cerr << "Error: size of vocabulary file " << vocab.size() << " != --vocab_size " << vocab_size << endl;
      }
    }
    // else, set it to the size of vocabulary read from file
    else {
      vocab_size = vocab.size();
    }

  }
  /*
  // construct vocabulary to contain top <vocab_size> most frequent words; all other words replaced by <unk>
  else {
  vocab.insert_word("<s>");
  vocab.insert_word("</s>");
  vocab.insert_word("<null>");

  // warn user that if --numberize is not set, there will be no vocabulary!
  if (!numberize) {
  cerr << "Warning: with --numberize 0 and --words_file == "", there will be no vocabulary!" << endl;
  }
  unordered_map<string,int> count;
  for (int i=0; i<train_data.size(); i++) {
  for (int j=0; j<train_data[i].size(); j++) {
  count[train_data[i][j]] += 1;
  }
  }

  vocab.insert_most_frequent(count, vocab_size);
  if (vocab.size() < vocab_size) {
  cerr << "warning: fewer than " << vocab_size << " types in training data; the unknown word will not be learned" << endl;
  }
  }
  */

  // write vocabulary to file
  if (write_words_file != "") {
    cerr << "Writing vocabulary to " << write_words_file << endl;
    writeWordsFile(vocab.words(), write_words_file);
  }

  // Write out numberized n-grams
  if (train_file != "")
  {
    cerr << "Writing training data to " << train_file << endl;
    if (mmap_file == true) {
      writeMmapNgrams(train_text,
                      ngram_size,
                      vocab,
                      numberize,
                      add_start_stop,
                      ngramize,
                      train_file,
                      train_data_size,
                      num_tokens,
                      randomize);
    } else {
      writeNgrams(train_text,
                  ngram_size,
                  vocab,
                  numberize,
                  add_start_stop,
                  ngramize,
                  train_file,
                  train_data_size,
                  sent_weights,
                  output_sent_weights_text);
    }
  }
  if (validation_file != "")
  {
    cerr << "Writing validation data to " << validation_file << endl;
    writeNgrams(validation_data,
                ngram_size,
                vocab,
                numberize,
                add_start_stop,
                ngramize,
                validation_file);
  }
}
