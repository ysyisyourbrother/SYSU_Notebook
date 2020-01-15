#include "process.h"
extern void print();
extern void printchar();
extern void delay();

extern int semaGet();
extern void semaP();
extern void semaV();

extern int fork();
extern void wait();
extern int exit();

main() {
   int s, tmp;
   s = semaGet(0);
   print("\r\nUser: forking...\r\n");
   tmp = fork();
   if(tmp) {
	   while(1) {
		   semaP(s);
		   semaP(s);
		   if(be_change) {
			   print(words);
			   be_change = 0;
		   }
		   print("Father enjoy the fruit ");
		   printInt(fruit_disk);
		   print("\r\n");
		   fruit_disk = 0;
		}
   }
   else {
	   print("User: forking again...\r\n");
	   tmp = fork();
	   if(tmp) {
		   while(1) {
			   be_change = 1;
			   write("Father will live one year after anther forever!\r\n");
			   semaV(s);
			   delay();
		   }
	   }
	   else {
		   while(1) {
			   putfruit();
			   semaV(s);
			   delay();
		   }
	   }
   }
}

