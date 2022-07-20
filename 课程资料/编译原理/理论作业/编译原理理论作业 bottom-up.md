## 编译原理理论作业 bottom-up

Ex4.6.2

**问题：**为联系4.2.1中的增广文法构造SLR项集。计算这些项集的GOTO函数。给出这个文法的语法分析表。这个文法是SLR文法吗？

提取左公因子并消除左递归：

0. $S'\rightarrow S$
1. $S\rightarrow aB$
2. $B\rightarrow aBAB$
3. $B\rightarrow \epsilon$
4. $A\rightarrow +$
5. $A \rightarrow *$

可以得到LR(0)的自动机：

<img src="/Users/yeshy/Downloads/IMG_4F78C7479B44-1.jpeg" alt="IMG_4F78C7479B44-1" style="zoom: 50%;" />

$FOLLOW(S) = \{\$\}$  $FOLLOW(A) = \{a, +, *, \$\}$  $FOLLOW(B) = \{+, *, \$\}$

可以得到语法分析表：

| 状态 | $a$   | $+$   | $*$   | $\$$  | S     | A     | B     |
| ---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 0    | $S_2$ |       |       |       | $S_1$ |       |       |
| 1    |       |       |       | acc   |       |       | $S_3$ |
| 2    | $S_4$ | $r_3$ | $r_3$ | $r_3$ |       |       |       |
| 3    |       |       |       | $r_1$ |       |       |       |
| 4    | $S_4$ | $r_3$ | $r_3$ | $r_3$ |       |       | $S_5$ |
| 5    |       | $S_7$ | $S_8$ |       |       | $S_6$ |       |
| 6    | $S_4$ | $r_3$ | $r_3$ | $r_3$ |       |       | $S_9$ |
| 7    | $r_4$ | $r_4$ | $r_4$ | $r_4$ |       |       |       |
| 8    | $r_5$ | $r_5$ | $r_5$ | $r_5$ |       |       |       |
| 9    |       | $r_2$ | $r_2$ | $r_2$ |       |       |       |

无冲突，是SLR文法



EX4.6.3利用练习4.6.2得到的语法分析表，给出处理输入$aa*a+$时的各个动作

| STACK     | SYMBOLS    | INPUT     | ACTION                        |
| --------- | ---------- | --------- | ----------------------------- |
| 0         |            | $aa*a+\$$ | 移入                          |
| 02        | a          | $a*a+\$$  | 移入                          |
| 024       | aa         | $*a+\$$   | 用$B\rightarrow \epsilon$归约 |
| 0245      | aaB        | $*a+\$$   | 移入                          |
| 02458     | aaB*       | $a+\$$    | 用$A\rightarrow *$ 归约       |
| 02456     | aaBA       | $a+\$$    | 移入                          |
| 024564    | $aaBAa$    | $+\$$     | 用$B\rightarrow \epsilon$归约 |
| 0245645   | $aaBAaB$   | $+\$$     | 移入                          |
| 02456457  | $aaBAaB+$  | $\$$      | 用$A\rightarrow +$ 归约       |
| 02456456  | $aaBAaBA$  | $\$$      | 用$B\rightarrow \epsilon$归约 |
| 024564569 | $aaBAaBAB$ | $\$$      | 用$B\rightarrow aBAB$归约     |
| 024569    | $aaBAB$    | $\$$      | 用$B\rightarrow aBAB$归约     |
| 023       | $aB$       | $\$$      | 用$S\rightarrow aB$归约       |
| 01        | $S$        | $\$$      | 接受                          |



