#include <algorithm>
#include <fstream>

#include <boost/algorithm/string/join.hpp>
//#include <boost/thread/thread.hpp>
#include <tclap/CmdLine.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "param.h"

#include "neuralLM.h"

using namespace std;
using namespace boost;
using namespace TCLAP;
using namespace Eigen;

using namespace nplm;

void score(neuralLM &lm, int minibatch_size, vector<int>& start, vector< vector<int> > &ngrams,
	   vector<double> &out) {
    if (ngrams.size() == 0) return;
    int ngram_size = ngrams[0].size();

    if (minibatch_size == 0)
    {
        // Score one n-gram at a time. This is how the LM would be queried from a decoder.
        for (int sent_id=0; sent_id<start.size()-1; sent_id++)
	{	  
	    double sent_log_prob = 0.0;
	    for (int j=start[sent_id]; j<start[sent_id+1]; j++) 
	        sent_log_prob += lm.lookup_ngram(ngrams[j]);
	    out.push_back(sent_log_prob);
	}
    }
    else
    {
	// Score a whole minibatch at a time.
        Matrix<double,1,Dynamic> log_probs(ngrams.size());

        Matrix<int,Dynamic,Dynamic> minibatch(ngram_size, minibatch_size);
	minibatch.setZero();
        for (int test_id = 0; test_id < ngrams.size(); test_id += minibatch_size)
	{
	    int current_minibatch_size = minibatch_size<ngrams.size()-test_id ? minibatch_size : ngrams.size()-test_id;
	    for (int j=0; j<current_minibatch_size; j++)
	        minibatch.col(j) = Map< Matrix<int,Dynamic,1> > (ngrams[test_id+j].data(), ngram_size);
	    lm.lookup_ngram(minibatch.leftCols(current_minibatch_size), log_probs.middleCols(test_id, current_minibatch_size));
	}

	for (int sent_id=0; sent_id<start.size()-1; sent_id++)
	{
	    double sent_log_prob = 0.0;
	    for (int j=start[sent_id]; j<start[sent_id+1]; j++)
	        sent_log_prob += log_probs[j];
	    out.push_back(sent_log_prob);
	}
    }
}

int main (int argc, char *argv[]) 
{
    param myParam;
    bool normalization;
    bool numberize, ngramize, add_start_stop;

    try {
      // program options //
      CmdLine cmd("Tests a two-layer neural probabilistic language model.", ' ' , "0.1");

      ValueArg<int> num_threads("", "num_threads", "Number of threads. Default: maximum.", false, 0, "int", cmd);
      ValueArg<int> minibatch_size("", "minibatch_size", "Minibatch size. Default: none.", false, 0, "int", cmd);

      ValueArg<bool> arg_ngramize("", "ngramize", "If true, convert lines to ngrams. Default: true.", false, true, "bool", cmd);
      ValueArg<bool> arg_numberize("", "numberize", "If true, convert words to numbers. Default: true.", false, true, "bool", cmd);
      ValueArg<bool> arg_add_start_stop("", "add_start_stop", "If true, prepend <s> and append </s>. Default: true.", false, true, "bool", cmd);

      ValueArg<bool> arg_normalization("", "normalization", "Normalize probabilities. 1 = yes, 0 = no. Default: 0.", false, 0, "bool", cmd);

      ValueArg<string> arg_test_file("", "test_file", "Test file (one tokenized sentence per line).", true, "", "string", cmd);

      ValueArg<string> arg_model_file("", "model_file", "Language model file.", true, "", "string", cmd);

      cmd.parse(argc, argv);

      myParam.model_file = arg_model_file.getValue();
      myParam.test_file = arg_test_file.getValue();

      normalization = arg_normalization.getValue();
      numberize = arg_numberize.getValue();
      ngramize = arg_ngramize.getValue();
      add_start_stop = arg_add_start_stop.getValue();

      myParam.minibatch_size = minibatch_size.getValue();
      myParam.num_threads = num_threads.getValue();

      cerr << "Command line: " << endl;
      cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << endl;
      
      const string sep(" Value: ");
      cerr << arg_test_file.getDescription() << sep << arg_test_file.getValue() << endl;
      cerr << arg_model_file.getDescription() << sep << arg_model_file.getValue() << endl;

      cerr << arg_normalization.getDescription() << sep << arg_normalization.getValue() << endl;
      cerr << arg_ngramize.getDescription() << sep << arg_ngramize.getValue() << endl;
      cerr << arg_add_start_stop.getDescription() << sep << arg_add_start_stop.getValue() << endl;

      cerr << minibatch_size.getDescription() << sep << minibatch_size.getValue() << endl;
      cerr << num_threads.getDescription() << sep << num_threads.getValue() << endl;
    }
    catch (TCLAP::ArgException &e)
    {
      cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
      exit(1);
    }

    myParam.num_threads = setup_threads(myParam.num_threads);

    ///// Create language model

    neuralLM lm;
    lm.read(myParam.model_file);
    lm.set_normalization(normalization);
    lm.set_log_base(10);
    lm.set_cache(1048576);
    int ngram_size = lm.get_order();
    int minibatch_size = myParam.minibatch_size;
    if (minibatch_size)
        lm.set_width(minibatch_size);

    ///// Read test data

    ifstream test_file(myParam.test_file.c_str());
    if (!test_file)
    {
	cerr << "error: could not open " << myParam.test_file << endl;
	exit(1);
    }
    string line;

    vector<int> start;
    vector<vector<int> > ngrams;

    while (getline(test_file, line))
    {
        vector<string> words;
        splitBySpace(line, words);

	vector<vector<int> > sent_ngrams;
	preprocessWords(words, sent_ngrams, ngram_size, lm.get_vocabulary(), numberize, add_start_stop, ngramize);

	start.push_back(ngrams.size());
	copy(sent_ngrams.begin(), sent_ngrams.end(), back_inserter(ngrams));
    }
    start.push_back(ngrams.size());

    int num_threads = 1;
    vector< vector<double> > sent_log_probs(num_threads);

    /*
    // Test thread safety
    boost::thread_group tg;
    for (int t=0; t < num_threads; t++) {
      tg.create_thread(boost::bind(score, lm, minibatch_size, boost::ref(start), boost::ref(ngrams), boost::ref(sent_log_probs[t]))); // copy lm
    }
    tg.join_all();
    */
    score(lm, minibatch_size, start, ngrams, sent_log_probs[0]);

    vector<double> log_likelihood(num_threads);
    std::fill(log_likelihood.begin(), log_likelihood.end(), 0.0);
    for (int i=0; i<sent_log_probs[0].size(); i++) {
        for (int t=0; t<num_threads; t++)
	    cout << sent_log_probs[t][i] << "\t";
	cout << endl;
        for (int t=0; t<num_threads; t++)
	log_likelihood[t] += sent_log_probs[t][i];
    }
    
    cerr << "Test log10-likelihood: ";
    for (int t=0; t<num_threads; t++)
      cerr << log_likelihood[t] << " ";
    cerr << endl;
    #ifdef USE_CHRONO
    cerr << "Propagation times:";
    for (int i=0; i<timer.size(); i++)
      cerr << " " << timer.get(i);
    cerr << endl;
    #endif
    
}
