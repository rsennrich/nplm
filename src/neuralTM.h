#ifndef NEURALTM_H
#define NEURALTM_H

#include <vector>
#include <cctype>
#include <cstdlib>
#include <boost/shared_ptr.hpp>

#include <../3rdparty/Eigen/Dense>

#include "util.h"
#include "vocabulary.h"
#include "neuralNetwork.h"

namespace nplm
{

class neuralTM : public neuralNetwork
{
    char map_digits;
    boost::shared_ptr<vocabulary> input_vocab, output_vocab;
    int start, null;

public:
    neuralTM() 
      : neuralNetwork(),
        map_digits(0),
        input_vocab(new vocabulary()),
        output_vocab(new vocabulary())
    { 
    }

    void set_map_digits(char value) { map_digits = value; }

    void set_input_vocabulary(const vocabulary &vocab)
    {
        *(this->input_vocab) = vocab;
        start = vocab.lookup_word("<s>");
        null = vocab.lookup_word("<null>");
    }

    void set_output_vocabulary(const vocabulary &vocab)
    {
        *(this->output_vocab) = vocab;
    }

    const vocabulary &get_input_vocabulary() const { return *(this->input_vocab); }
    const vocabulary &get_output_vocabulary() const { return *(this->output_vocab); }

    int lookup_input_word(const std::string &word) const
    {
        if (map_digits)
	    for (int i=0; i<word.length(); i++)
	        if (isdigit(word[i]))
		{
		    std::string mapped_word(word);
		    for (; i<word.length(); i++)
		        if (isdigit(word[i]))
			    mapped_word[i] = map_digits;
		    return input_vocab->lookup_word(mapped_word);
		}
        return input_vocab->lookup_word(word);
    }

    int lookup_output_word(const std::string &word) const
    {
        if (map_digits)
	    for (int i=0; i<word.length(); i++)
	        if (isdigit(word[i]))
		{
		    std::string mapped_word(word);
		    for (; i<word.length(); i++)
		        if (isdigit(word[i]))
			    mapped_word[i] = map_digits;
		    return output_vocab->lookup_word(mapped_word);
		}
	return output_vocab->lookup_word(word);
    }

    double lookup_ngram(const int *ngram_a, int n)
    {
        Eigen::Matrix<int,Eigen::Dynamic,1> ngram(m->ngram_size);
	for (int i=0; i<m->ngram_size; i++)
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
        std::vector<std::string> input_words;
        std::vector<std::string> output_words;
        m->read(filename, input_words, output_words);
        set_input_vocabulary(vocabulary(input_words));
        set_output_vocabulary(vocabulary(output_words));
        resize();
	// this is faster but takes more memory
        //m->premultiply();
    }

};

} // namespace nplm

#endif
