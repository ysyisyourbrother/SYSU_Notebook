#include "process.h"
extern int fork();
extern void wait();
extern int exit();

char str[80] = "129djwqhdsajd128dw9i39ie93i8494urjoiew98kdkd";
int letterNr = 0;
int ex;
void main()
{
    int pid=0;
    int i;
    print("The string is: ");
    print(str);
    print("\r\n");
    pid = fork(); 
    /*printchar(pid+'0');
    print("\n\r");*/ 
    if (pid == -1)
    {
        print("error in fork!");
        exit(-1);
    }
    /* fork进程传回来的pid是大于0的整数 即子进程的id */
    if (pid)
    {
        print("Father process: This is the father process.\r\n");
        print("Father process: My subprocess's ID is: ");
        printnumber(pid);
        print("\r\n");
        wait();/* 等待子进程运行 */
        print("Father process: The numbers of letters in string is: ");
        printnumber(44);
        print("\n\r");
        delay();
        exit(0);
    }
    /* 子进程传回来的fork是0 执行以下函数 */
    else
    {
        print("Subprocess: This is the subprocess.\r\n");
        for (i=0; str[i]; ++i)
            letterNr++;
        print("Subprocess: The result that subprocess get after calculating is ");
        printnumber(letterNr);
        print("\r\n");
        exit(0);
    }
}