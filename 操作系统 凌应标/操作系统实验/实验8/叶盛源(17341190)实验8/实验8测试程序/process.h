#ifndef PROCESS
#define PROCESS

extern void printchar();

char temp[1000];
char words[100];
int be_change = 0, fruit_disk, rand = 0;

void write(char *p) {
	int i = 0;
	while(*p != '\0') {
		words[i++] = *p;
		p++;
	}
}

void putfruit() {
	fruit_disk = rand++ % 10;
}

/* 循环输出长的字符串 */
void print(char* s)
{
    while (*s)
    {
        printchar(*s);
        s++;
    }
}

void reverse(char str[],int len) {
	int i;
    char t_char[100];
	for(i = 0;i < len;++i) {
		t_char[i] = str[len-i-1];
	}
	for(i = 0;i < len;++i) {
		str[i] = t_char[i];
	}
}

void printInt(int ans) {
	int i = 0;
    char output[100];
	if(ans == 0) {
		output[0] = '0';
		i++;
	}
	while(ans) {
		int t = ans%10;
		output[i++] = '0'+t;
		ans/=10;
	}
	reverse(output,i);
	output[i] = '\0';
	print(output);
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