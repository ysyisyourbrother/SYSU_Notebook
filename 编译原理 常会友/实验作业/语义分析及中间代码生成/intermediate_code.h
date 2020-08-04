#ifndef INTERMEDIATE_CODE_H
#define INTERMEDIATE_CODE_H

#include "datatype.h"

class IntermediateCode {
public:
	int label_num = 0;
	int var_num = 1;
	vector<string> res_;

	void stmt_seq_node(TreeNode* root);

	void stmt_node(TreeNode* root);

	void if_node(TreeNode* root, string end_label);

	void while_node(TreeNode* root, string end_label);

	void read_node(TreeNode* root);

	void write_node(TreeNode* root);

	void repeat_node(TreeNode* root, string end_label);

	void assign_node(TreeNode* root);

	void cond_exp(TreeNode* root, string true_label, string false_label);

	string log_or(TreeNode* root);

	string log_and(TreeNode* root);

	string comparison_exe(TreeNode* root);

	string add_exp(TreeNode* root);

	string mul_exe(TreeNode* root);

	string factor(TreeNode* root);


	string get_new_label();

	string get_var_name();
};

#endif