#include <iostream>
#include <vector>
#include <fstream>

#include <boost/algorithm/string/join.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/interprocess/containers/vector.hpp>

#include <tclap/CmdLine.h>

#include "neuralLM.h"
#include "util.h"

// take ngramized and numberized file (ready for trainNeuralNetwork)
// and create mmap_file (for training without loading all data into memory)

using namespace TCLAP;
using namespace boost;
using namespace nplm;
namespace ip = boost::interprocess;

typedef ip::allocator<int, ip::managed_mapped_file::segment_manager> intAllocator;
typedef ip::vector<int, intAllocator> vec;
typedef ip::allocator<vec, ip::managed_mapped_file::segment_manager> vecAllocator;

typedef long long int data_size_t; // training data can easily exceed 2G instances

data_size_t getNumLines(const string &filename) {
  ifstream training(filename.c_str());
  data_size_t lines = 0;
  std::string line;
  while (std::getline(training, line)) {
    if ((lines%100000)==0) {
        std::cerr<<lines<<"...";
    }
    ++lines;
  }
  training.close();
  return lines;
}

int getNgramSize(const string &filename) {
  ifstream training(filename.c_str());
  std::string line;
  std::getline(training, line);
  std::vector<std::string> ngram;
  splitBySpace(line, ngram);
  training.close();
  return ngram.size();
}

void writeMmap(const string &filename_input,
          const string &filename_output,
          int ngram_size,
          data_size_t num_tokens) {

    // Open the memory mapped file and create the allocators
    ip::managed_mapped_file mfile(ip::create_only,
        filename_output.c_str(),
        num_tokens*ngram_size*sizeof(int)+1024UL*1024UL);
    intAllocator ialloc(mfile.get_segment_manager());
    vecAllocator valloc (mfile.get_segment_manager());

    vec *mMapVec= mfile.construct<vec>("vector")(num_tokens*ngram_size,0,ialloc);

    std::cerr<<"The size of mmaped vec is "<<mMapVec->size() << std::endl;

  ifstream training(filename_input.c_str());
  data_size_t i = 0;
  std::string line;
  std::vector<std::string> ngram;
  while (std::getline(training, line)) {

    if ((i%100000)==0) {
        std::cerr<<i<<"...";
    }

    splitBySpace(line, ngram);
    if (ngram.size() != ngram_size)
    {
        std::cerr << "Error: expected " << ngram_size << " fields in instance, found " << ngram.size() << std::endl;
        std::exit(-1);
    }

    for (int j=0; j<ngram_size; j++) {
      mMapVec->at(i*ngram_size+j) = boost::lexical_cast<int>(ngram[j]);
    }

    ++i;
  }

  training.close();
  ip::managed_mapped_file::shrink_to_fit(filename_output.c_str());
 
}


int main(int argc, char *argv[])
{
  ios::sync_with_stdio(false);
  int ngram_size;
  data_size_t num_tokens;

  std::string input_file, output_file;


  try
  {
    CmdLine cmd("take ngramized and numberized file and create memory mapped file (for training without loading add training data into memory).", ' ', "0.1");

    // The options are printed in reverse order
    ValueArg<std::string> arg_output_file("", "output_file", "Output training data (memory mapped file).", true, "", "string", cmd);
    ValueArg<std::string> arg_input_file("", "input_file", "Input training data (numberized n-grams).", true, "", "string", cmd);

    cmd.parse(argc, argv);

    input_file = arg_input_file.getValue();
    output_file = arg_output_file.getValue();

    std::cerr << "Command line: " << std::endl;
    std::cerr << boost::algorithm::join(std::vector<std::string>(argv, argv+argc), " ") << std::endl;

    const std::string sep(" Value: ");
    std::cerr << arg_input_file.getDescription() << sep << arg_input_file.getValue() << std::endl;
    std::cerr << arg_output_file.getDescription() << sep << arg_output_file.getValue() << std::endl;
  }
  catch (TCLAP::ArgException &e)
  {
    std::cerr << "error: " << e.error() <<  " for arg " << e.argId() << std::endl;
    std::exit(1);
  }

  std::cerr << "counting number of lines:" << std::endl;
  ngram_size = getNgramSize(input_file);
  num_tokens = getNumLines(input_file);
  std::cerr << std::endl;
  std::cerr << "writing mmap file:" << std::endl;
  writeMmap(input_file, output_file, ngram_size, num_tokens);
  std::cerr << std::endl;

}
