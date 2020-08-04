## 编译原理理论作业 top-down

Exercise 4.2.1

考虑上下文无关文法
$$
S\rightarrow S\ S\ +\ |S\ S\ *\ |\ a
$$


以及串$aa+a\ *$

2. 给出这个串的一个最右推导：
   $$
   S \rightarrow S\ S\ *\rightarrow S\ a\ *\rightarrow S\ S+a\ *\rightarrow S\ a+a\ *\rightarrow a\ a+a\ *
   $$

3. 给出这个串的一颗语法分析树

   <img src="assets/编译原理理论作业 top-down/IMG_A7029F6652EF-1.jpeg" alt="IMG_A7029F6652EF-1" style="zoom:50%;" />

4. 这个文法是否为二义性？

   无，理由如下。

   按右递归分解，若当前最右字符为+。则只能由$S \rightarrow SS+$得到

   若当前最右字符为a，只能由$S \rightarrow a$得到

   若为$*$，只能从$S \rightarrow SS$*得到

   因此不存在没有二义性



Exercise4.4.1

为下面每一个文法设计一个预测分析器，并给出预测分析表，你可能先要对文法进行提取左公因子或消除左递归的操作

2. $S \rightarrow +SS|*SS|a$ with string $+*aaa$

   |      | +                 | *                  | a                |
   | ---- | ----------------- | ------------------ | ---------------- |
   | S    | $S\rightarrow+SS$ | $S\rightarrow *SS$ | $S\rightarrow a$ |

3. $S \rightarrow S(S)S|\epsilon$  with string (( ) ( ))

$$
S\rightarrow S'\\
S' \rightarrow (S)SS'|\epsilon
$$

|      | (                                              | )                        | $                       |
| ---- | ---------------------------------------------- | ------------------------ | ----------------------- |
| S    | $S\rightarrow S'$                              | $S\rightarrow S'$        | $S\rightarrow S'$       |
| S'   | $S'\rightarrow (S)SS'\\S'\rightarrow \epsilon$ | $S'\rightarrow \epsilon$ | $S\rightarrow \epsilon$ |



4. $S\rightarrow S+S|SS|(S)｜S*｜a$  with string (a+a)*a

   提取左因式
   $$
   S\rightarrow SA|(S)|a\\
   A\rightarrow +S|S|*
   $$
   消除递归
   $$
   S\rightarrow(S)S'|aS'\\
   S'\rightarrow AS'|\epsilon\\
   A\rightarrow +S|(S)S'|aS'|*
   $$

   |      | (                                          | )                       | +                                          | *                                          | a                                          | $                       |
   | ---- | ------------------------------------------ | ----------------------- | ------------------------------------------ | ------------------------------------------ | ------------------------------------------ | ----------------------- |
   | S    | $S\rightarrow(S)S'$                        |                         |                                            |                                            | $S\rightarrow aS'$                         |                         |
   | S'   | $S'\rightarrow AS'\\S'\rightarrow\epsilon$ | $S'\rightarrow\epsilon$ | $S'\rightarrow AS'\\S'\rightarrow\epsilon$ | $S'\rightarrow AS'\\S'\rightarrow\epsilon$ | $S'\rightarrow AS'\\S'\rightarrow\epsilon$ | $S'\rightarrow\epsilon$ |
   | A    | $A\rightarrow(S)S'$                        |                         | $A\rightarrow+S$                           | $A\rightarrow*$                            | $A\rightarrow aS'$                         |                         |

   