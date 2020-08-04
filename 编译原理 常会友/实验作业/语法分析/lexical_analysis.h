#pragma once
#ifndef LEXICAL_ANALYSIS_H
#define LEXICAL_ANALYSIS_H

#include "datatype.h"


class LexicalAnalysis {
private:
	enum Status {
		blank,
		number,
		letter,
		single_quote,
		special_char,
		error,
		comment,
	};
	enum CharType {
		c_blank,
		c_number,
		c_letter,
		c_single_quote,
		c_special_char,
		c_newline,
		c_illegal,
	};

	bool is_blank(char c);

	bool is_newline(char c);

	bool is_number(char c);

	bool is_letter(char c);

	bool is_special_symbol(char c);

	bool is_single_quote(char c);

	CharType get_char_type(char c);

public:
	vector<ErrorMsg> AnalyseTokens(ifstream& sourceCode, Tokens& tokens);
	void run(ErrorMsgs& error_msgs, Tokens& tokens);
};

#endif
