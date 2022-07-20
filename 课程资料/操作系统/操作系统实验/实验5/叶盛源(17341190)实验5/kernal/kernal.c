extern void cls();
extern char Readchar();
extern void printChar();
extern void RunProm();
extern void loadListFromDisk();
extern void loadbatchFromDisk();
extern void readList();
extern void readbatch();
extern void loadUser6FromDisk();

int flag;
int leave_batch;
char in;
char * str;
char command[100];
void printf(char* str)
{
    int i=0;
    while(str[i])
    {
        printChar(str[i]);
        i++;
    }
}
int if_equal(char* a,char* b)
{
    int i=0,j=0;
    while(a[i]!=0&&b[j]!=0)
    {
        if(a[i]!=b[j])
            return 0;
        i++;
        j++;
    }
    if(a[i]==0&&b[j]==0)
        return 1;
    else
        return 0;
}
void Readcommand(char* command)
{
    int i=0;
    Readchar();
    while(in!=13)
    {
        if(in==8)
        {
            if(i==0) 
            {
                printChar(in);
                printChar('>');
                Readchar();
                continue;
            }
            printChar(in);
            printChar(32);
            i--;
            printChar(in);
            Readchar();
            continue;
        }
        printChar(in);
        command[i]=in;  
        i++;
        Readchar();
    }
    command[i]='\0';
}

void Menu()
{
    printf("=================================================================\n\r");
    printf("|                       Welcome to my OS!                       |\n\r");
    printf("|                     Build on March 25 2018                    |\n\r");
    printf("|          Author:Ye Shengyuan & Student Number:17341190        |\n\r");
    printf("=================================================================\n\r");
    printf("Please enter instruction below. Enter -h to view instruction list\n\r");
}
void help()
{
    printf("\n\r-h:view instruction list\n\r");
    printf("-cls:clear the display\n\r");
    printf("-p1:run the user program 1\n\r");
    printf("-p2:run the user program 2\n\r");
    printf("-p3:run the user program 3\n\r");
    printf("-p4:run the user program 4\n\r");
    printf("-p5:run the user program 5\n\r");
    printf("-ls:show the file list\n\r");
    printf("-b:begin processing batch\n\r");
    printf("-q:exit the operating system\n\r");
}
void list()
{
    char name[10];
    char size[10];
    int sector;
    int i=1;
    int j=0;
    int x;
    int count;
    readList(0);
    count=str[0];
    name[9]='\0';
    size[9]='\0';
    printf("\n\rTotal Files:");
    printChar(count+'0');
    printf("\n\r FileName     FileSize\n\r");
    while(count--)
    {
        j=0;
        for(x=0;x<9;x++)
        {
            name[x]=32;
            size[x]=32;
        }
        while(str[i]!=0)
        {
            name[j]=str[i];
            j++;
            i++;
        }
        i++,j=0;
        while(str[i]!=0)
        {
            size[j]=str[i];
            j++;
            i++;
        }
        size[j]='B';
        i=i+3;
        printChar(6-count-1+'0');
        printf(".");
        printf(name);
        printf("   ");
        printf(size);
        printf("\n\r"); 
    }
}
void runcommand(char * com)
{
    if(if_equal(com,"-h"))
    {
        help();
    }
    else if(if_equal(com,"-cls"))
    {
        cls();
    }
    else if(if_equal(com,"-p1"))
    {
        cls();
        readList(14);
        RunProm((*str));
        cls();
        
    }
    else if(if_equal(com,"-p2"))
    {
        cls();
        readList(26);
        RunProm((*str));
        cls();
    }
    else if(if_equal(com,"-p3"))
    {
        cls();
        readList(39);
        RunProm((*str));
        cls();
    }
    else if(if_equal(com,"-p4"))
    {
        cls();
        readList(55);
        RunProm((*str));
        cls();
    }
    else if(if_equal(com,"-p5"))
    {
        cls();
        readList(69);
        RunProm((*str));
        cls();
    }
    else if(if_equal(com,"-b"))
    {
        cls();
        printf("please enter your instructions queue.\n\rPlease separate instructions by Spaces.Enter -qb to leave batch processing.\n\r");
        Readcommand(command);
        batch(command);
    }
    else if(if_equal(com,"-ls"))
    {
        list();
    }
    else if(if_equal(com,"-q"))
    {
        flag=0;
    }
    else if(if_equal(com,"-qb"))
    {
        leave_batch=0;
    }
    else {
        printf("\r\nPlease enter correct instructions");
    }
}


batch(char* com){
    int i;
    int j;
    char subcommand[10];
    flag=1;
    i=0;
    j=0;
    leave_batch=1;
    while(com[i]!='\0'&&flag&&leave_batch)
    {
        if(com[i]!=32)
        {
            subcommand[j]=com[i];
            i++;
            j++;
        }
        else
        {
            subcommand[j]='\0';
            runcommand(subcommand);
            i++;
            j=0;
        }
    }
    if(com[i]=='\0')
    {
        subcommand[j]='\0';
        runcommand(subcommand);
    }
}

cmain(){
    loadListFromDisk();
    loadbatchFromDisk();
    readbatch();
    batch(str);
    cls();
    Menu();
    flag=1;
    while(flag)
    {  
        printf("\n\r>>");
        Readcommand(command);
        runcommand(command);
    }
    cls();
    printf("\n\rThank you for using!\n\rBye-Bye!\n\r");
    Readchar();
}

