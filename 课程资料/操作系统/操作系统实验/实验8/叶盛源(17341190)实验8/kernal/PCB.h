extern void stackcopy(int, int);
extern void Switch();
extern void printChar();


int NEW = 0;
int READY = 1;
int RUNNING = 2;
int BLOCKED = 3;
int EXIT = 4;

#define MAX_PCB_NUMBER 8/*最多同时放这么多个进程 */

typedef struct RegisterImage{
	int SS;
	int GS;
	int FS;
	int ES;
	int DS;
	int DI;
	int SI;
	int BP;
	int SP;
	int BX;
	int DX;
	int CX;
	int AX;
	int IP;
	int CS;
	int FLAGS;
}RegisterImage;

typedef struct PCB{
	RegisterImage regImg;
	int Process_Status;
	int Used; /* 查看这个块是否被使用了 */
	int FatherID;
}PCB;


PCB pcb_list[MAX_PCB_NUMBER];
int CurrentPCBno = 0; /*当前正在运行的进程编号 */
int Program_Num = 0;/* */
int Segment = 0x2000;


extern void printChar();

PCB* Current_Process();
void Save_Process(int,int, int, int, int, int, int, int,
		  int,int,int,int, int,int, int,int );
void init(PCB*, int, int);
void Schedule();
void special();
void create_new_PCB();
int do_fork();
void delay() /* 延时函数 */
{
    int i;
    int j;
    for(i=0; i<20000; i++)
        for(j=0; j<20000; j++);
    return ;
}

void Save_Process(int gs,int fs,int es,int ds,int di,int si,int bp,
		int sp,int dx,int cx,int bx,int ax,int ss,int ip,int cs,int flags)
{
	pcb_list[CurrentPCBno].regImg.AX = ax;
	pcb_list[CurrentPCBno].regImg.BX = bx;
	pcb_list[CurrentPCBno].regImg.CX = cx;
	pcb_list[CurrentPCBno].regImg.DX = dx;

	pcb_list[CurrentPCBno].regImg.DS = ds;
	pcb_list[CurrentPCBno].regImg.ES = es;
	pcb_list[CurrentPCBno].regImg.FS = fs;
	pcb_list[CurrentPCBno].regImg.GS = gs;
	pcb_list[CurrentPCBno].regImg.SS = ss;

	pcb_list[CurrentPCBno].regImg.IP = ip;
	pcb_list[CurrentPCBno].regImg.CS = cs;
	pcb_list[CurrentPCBno].regImg.FLAGS = flags;
	
	pcb_list[CurrentPCBno].regImg.DI = di;
	pcb_list[CurrentPCBno].regImg.SI = si;
	pcb_list[CurrentPCBno].regImg.SP = sp;
	pcb_list[CurrentPCBno].regImg.BP = bp;
}

void Schedule()
{   
    int End = CurrentPCBno;
    if (pcb_list[CurrentPCBno].Process_Status == RUNNING)
        pcb_list[CurrentPCBno].Process_Status = READY;
    CurrentPCBno += 1;
    if (CurrentPCBno >= MAX_PCB_NUMBER)
            CurrentPCBno = 1;
    while (CurrentPCBno != End)
    {
		/*如果这个PCB块被进程使用了 */
        if (pcb_list[CurrentPCBno].Used == 1)
        {
            if (pcb_list[CurrentPCBno].Process_Status == READY)
            {
                pcb_list[CurrentPCBno].Process_Status = RUNNING;
                return ;
            }
            else if (pcb_list[CurrentPCBno].Process_Status == NEW)/* 如果是new状态的话要单独讨论一次 */
                return ;
        }
		/* 如果没有被使用就继续往下找 */
        CurrentPCBno++;
        if (CurrentPCBno >= MAX_PCB_NUMBER)
            CurrentPCBno = 1;
    }
	/* 如果没有其他可以被调度的进程了，就要恢复刚刚才运行进程 如果它也结束了运行就返回内核 */
    if (pcb_list[CurrentPCBno].Used == 1 && pcb_list[CurrentPCBno].Process_Status == READY)
        pcb_list[CurrentPCBno].Process_Status = RUNNING;
    else 
        CurrentPCBno = 0;
}

PCB* Current_Process(){

	return &pcb_list[CurrentPCBno];
}

void init(PCB* pcb,int segement, int offset)
{
	pcb->Used=0;
	pcb->FatherID=0;
	pcb->regImg.GS = 0xb800;/*显存*/
	pcb->regImg.SS = segement;
	pcb->regImg.ES = segement;
	pcb->regImg.DS = segement;
	pcb->regImg.CS = segement;
	pcb->regImg.FS = segement;
	pcb->regImg.IP = offset;
	pcb->regImg.SP = offset - 4;
	pcb->regImg.AX = 0;
	pcb->regImg.BX = 0;
	pcb->regImg.CX = 0;
	pcb->regImg.DX = 0;
	pcb->regImg.DI = 0;
	pcb->regImg.SI = 0;
	pcb->regImg.BP = 0;
	pcb->regImg.FLAGS = 512;
	pcb->Process_Status = NEW;
}

void special()
{
	if(pcb_list[CurrentPCBno].Process_Status==NEW)
		pcb_list[CurrentPCBno].Process_Status=RUNNING;
}

/* 
	实验七新增部分代码 
*/
/*遍历PCB表并找到一个无进程的空PCB块*/
int Findempty()
{
    int index = 1;
    while (index < MAX_PCB_NUMBER)
    {
        if (pcb_list[index].Used != 1)
            return index;
        index++;
    }
    return -1;
}

void PCBcopy(PCB* p1, PCB* p2)
{
    p1->regImg.AX = p2->regImg.AX;
    p1->regImg.BX = p2->regImg.BX;
    p1->regImg.CX = p2->regImg.CX;
    p1->regImg.DX = p2->regImg.DX;
    p1->regImg.CS = p2->regImg.CS;
    p1->regImg.IP = p2->regImg.IP;
    p1->regImg.DS = p2->regImg.DS;
    p1->regImg.ES = p2->regImg.ES;
    p1->regImg.GS = p2->regImg.GS;
    p1->regImg.FS = p2->regImg.FS;
	/* 因为子进程和父进程不共享栈，所以栈指针不同，栈的内容另外拷贝 */
    /*p1->regImg.SS = p2->regImg.SS;*/
    p1->regImg.DI = p2->regImg.DI;
    p1->regImg.SI = p2->regImg.SI;
    p1->regImg.BP = p2->regImg.BP;
    p1->regImg.SP = p2->regImg.SP;
    p1->regImg.FLAGS = p2->regImg.FLAGS;
    p1->Process_Status = READY;
}

/*...Copy PCB...*/
int do_fork()
{
    int i = Findempty();
    if (i == -1){
    /* 没有空闲的PCB块 ax作为返回值 */
        pcb_list[CurrentPCBno].regImg.AX = -1;
        return ;
    }
    Program_Num++;  
	/* 拷贝 */
    PCBcopy(&pcb_list[i], &pcb_list[CurrentPCBno]);
    stackcopy(pcb_list[i].regImg.SS, pcb_list[CurrentPCBno].regImg.SS);
    pcb_list[i].FatherID = CurrentPCBno;
    pcb_list[i].Used =  1;
    pcb_list[i].regImg.AX = 0;/* 新创建的线程ax=0作为返回值 */
    pcb_list[CurrentPCBno].regImg.AX = i;/* 新创建的线程ax!=0作为返回值 */
    pcb_list[CurrentPCBno].Process_Status = READY;  
	/* 重新继续运行 */
    Switch();
}

void do_wait()
{
	/* 如果父进程要等待，则阻塞父进程，等待子进程运行完成 */
    pcb_list[CurrentPCBno].Process_Status = BLOCKED;
    Schedule();
    Switch();
}

int do_exit(int ch)
{
    int k;
    int FatherID = pcb_list[CurrentPCBno].FatherID;
    pcb_list[CurrentPCBno].Process_Status = EXIT;
	/* 结束此进程并重新初始化进程控制块 */
    init(&pcb_list[CurrentPCBno], (CurrentPCBno+1)*0x1000, 0x100);
	/* 如果当前退出进程的父进程不是内核，解除父进程的阻塞 */
    if (FatherID != 0){
		/* 唤醒父进程 */
        pcb_list[FatherID].Process_Status = READY;
		/* 用ax来传递信号 */
        pcb_list[FatherID].regImg.AX = ch;
    }
    Program_Num --;
    Segment=0x2000;
    for (k=0; k<15; ++k)
        delay();
    Schedule();
    Switch();
}

/*信号量部分*/
#define nrsemaphore 10
#define nrpcb 10

typedef struct semaphoretype {
    int count;
    int blocked_pcb[nrpcb];
    int used, front, tail;
} semaphoretype;

semaphoretype semaphorequeue[nrsemaphore]; /* 定义信号量的数组 */

/* 获取一个空闲的信号量并初始化它 */
int semaGet(int value) {
    int i = 0;
    while (semaphorequeue[i].used == 1 && i < nrsemaphore) { ++i; }
    if (i < nrsemaphore) {
        semaphorequeue[i].used = 1;
        semaphorequeue[i].count = value;
        semaphorequeue[i].front = 0;
        semaphorequeue[i].tail = 0;
		pcb_list[CurrentPCBno].regImg.AX = i;
		Switch();
		return i;
    }
	else {
		pcb_list[CurrentPCBno].regImg.AX = -1;
		Switch();
		return -1;
	}
}

void semaFree(int s) {
	semaphorequeue[s].used = 0;
}

void semaBlock(int s) {
	pcb_list[CurrentPCBno].Process_Status = BLOCKED;
	if ((semaphorequeue[s].tail + 1) % nrpcb == semaphorequeue[s].front) {
		return;
	}
	semaphorequeue[s].blocked_pcb[semaphorequeue[s].tail] = CurrentPCBno;
	semaphorequeue[s].tail = (semaphorequeue[s].tail + 1) % nrpcb;
}

void semaWakeUp(int s) {
	int t;
	if (semaphorequeue[s].tail == semaphorequeue[s].front) {
		return;
	}
	t = semaphorequeue[s].blocked_pcb[semaphorequeue[s].front];
	pcb_list[t].Process_Status = READY;
	semaphorequeue[s].front = (semaphorequeue[s].front + 1) % nrpcb;
}

/* 信号量的P操作，阻塞进程 */
void semaP(int s) {
	semaphorequeue[s].count--;
	if (semaphorequeue[s].count < 0) {
		semaBlock(s);
		Schedule();
	}
	Switch();
}

void semaV(int s) {
	semaphorequeue[s].count++;
	if (semaphorequeue[s].count <= 0) {
		semaWakeUp(s);
		Schedule();
	}
	Switch();
}

/*初始化信号量块 在主程序中调用*/
void initsema() {
	int i;
	for (i = 0; i < nrsemaphore; ++i) {
		semaphorequeue[i].used = 0;
		semaphorequeue[i].count = 0;
		semaphorequeue[i].front = 0;
		semaphorequeue[i].tail = 0;
	}
}