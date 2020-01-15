#ifndef PROCESS
#define PROCESS

extern void printchar();

char temp[1000];

/* 循环输出长的字符串 */
void print(char* s)
{
    while (*s)
    {
        printchar(*s);
        s++;
    }
}

/* 输出一个数字 */
void printnumber(int number)
{
    int i = 0;
    while (number)
    {
        temp[i++] = number%10 + '0';
        number /= 10;
    }
    i -= 1;
    while (i >= 0)
        printchar(temp[i--]);
}

void delay()
{
    int i;
    int j;
    for(i=0; i<10000; i++)
        for(j=0; j<10000; j++);
    return ;
}
#endif 