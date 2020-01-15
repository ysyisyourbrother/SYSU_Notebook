org  7c00h		; BIOS将把引导扇区加载到0:7C00h处，并开始执行
OffsetOfUserPrg equ 0A100h

Start:
	  mov	ax, cs	       ; 置其他段寄存器值与CS相同
	  mov	ds, ax	       ; 数据段
	  mov	bp, Message		 ; BP=当前串的偏移地址
	  mov	ax, ds		 ; ES:BP = 串地址
	  mov	es, ax		 ; 置ES=DS ;段寄存器之间不能直接传输，故要通过ax
    mov bx,0B800h				; 文本窗口显存起始地址
	  mov gs,bx					; GS = B800h

    ;使用int10h的清屏功能
    mov ah,6
    mov al,0
    mov ch,0  
    mov cl,0
    mov dh,24  
    mov dl,79
    mov bh,7 
    int 10h

	  mov	cx, MessageLength  ; CX = 串长
	  mov	ax, 1301h		 ; AH = 13h（功能号）、AL = 01h（光标置于串尾）
	  mov	bx, 0007h		 ; 页号为0(BH = 0) 黑底白字(BL = 07h)
    mov dx,0  ;开始输出的的行列号
	  int	10h			 ; BIOS的10h功能：显示一行字符


    ;使用int16指令输入选择跳转
    ;使用循环来确保输入实在1，2，3，4中的一个数字，
input:
    mov ah,0
    int 16h
    cmp al,'1'
    jz switch
    cmp al,'2'
    jz switch
    cmp al,'3'
    jz switch
    cmp al,'4'
    jz switch
    jmp input

switch:
    mov bl,'1'
    cmp al,bl
    jz program1
    mov bl,'2'
    cmp al,bl
    jz program2
    mov bl,'3'
    cmp al,bl
    jz program3
    mov bl,'4'
    cmp al,bl
    jz program4
    

program1:
     ;读软盘或硬盘上的若干物理扇区到内存的ES:BX处：
    mov ax,cs                ;段地址 ; 存放数据的内存基地址
    mov es,ax                ;设置段地址（不能直接mov es,段地址）
    mov bx, OffsetOfUserPrg           ;偏移地址; 存放数据的内存偏移地址,把磁盘的数据放在该偏移量地址里
    mov ah,2                 ;功能号，读磁盘
    mov al,1                 ;读入扇区数
    mov dl,0                 ;驱动器号 ; 软盘为0，硬盘和U盘为80H
    mov dh,0                 ;磁头号 ; 起始编号为0
    mov ch,0                 ;柱面号 ; 起始编号为0
    mov cl,2                 ;起始扇区号 ; 起始编号为1
    int 13H ;                调用读磁盘BIOS的13h功能
      ; 用户程序a.com已加载到指定内存区域中
    jmp OffsetOfUserPrg

program2:
     ;读软盘或硬盘上的若干物理扇区到内存的ES:BX处：
    mov ax,cs                ;段地址 ; 存放数据的内存基地址
    mov es,ax                ;设置段地址（不能直接mov es,段地址）
    mov bx, OffsetOfUserPrg           ;偏移地址; 存放数据的内存偏移地址
    mov ah,2                 ;功能号，读磁盘
    mov al,1                 ;读入扇区数
    mov dl,0                 ;驱动器号 ; 软盘为0，硬盘和U盘为80H
    mov dh,0                 ;磁头号 ; 起始编号为0
    mov ch,0                 ;柱面号 ; 起始编号为0
    mov cl,3                 ;起始扇区号 ; 起始编号为1
    int 13H ;                调用读磁盘BIOS的13h功能
      ; 用户程序a.com已加载到指定内存区域中
    jmp OffsetOfUserPrg

program3:
     ;读软盘或硬盘上的若干物理扇区到内存的ES:BX处：
    mov ax,cs                ;段地址 ; 存放数据的内存基地址
    mov es,ax                ;设置段地址（不能直接mov es,段地址）
    mov bx, OffsetOfUserPrg           ;偏移地址; 存放数据的内存偏移地址
    mov ah,2                 ;功能号，读磁盘
    mov al,1                 ;读入扇区数
    mov dl,0                 ;驱动器号 ; 软盘为0，硬盘和U盘为80H
    mov dh,0                 ;磁头号 ; 起始编号为0
    mov ch,0                 ;柱面号 ; 起始编号为0
    mov cl,4                 ;起始扇区号 ; 起始编号为1
    int 13H ;                调用读磁盘BIOS的13h功能
      ; 用户程序a.com已加载到指定内存区域中
    jmp OffsetOfUserPrg

program4:
     ;读软盘或硬盘上的若干物理扇区到内存的ES:BX处：
    mov ax,cs                ;段地址 ; 存放数据的内存基地址
    mov es,ax                ;设置段地址（不能直接mov es,段地址）
    mov bx, OffsetOfUserPrg           ;偏移地址; 存放数据的内存偏移地址
    mov ah,2                 ;功能号，读磁盘
    mov al,1                 ;读入扇区数
    mov dl,0                 ;驱动器号 ; 软盘为0，硬盘和U盘为80H
    mov dh,0                 ;磁头号 ; 起始编号为0
    mov ch,0                 ;柱面号 ; 起始编号为0
    mov cl,5                 ;起始扇区号 ; 起始编号为1
    int 13H ;                调用读磁盘BIOS的13h功能
      ; 用户程序a.com已加载到指定内存区域中
    jmp OffsetOfUserPrg

Message:
    db "Hello, MyOS is Ready!",0AH,0DH
    db "Here is the start Memu.Please Enter the serial number selection program: ",0AH,0DH
    db "1.Personal information",0AH,0DH
    db "2.stone",0AH,0DH
    db "3.square",0AH,0DH
    db "4.sand clock",0AH,0DH
    
    MessageLength  equ ($-Message)
    times 510-($-$$) db 0
    db 0x55,0xaa
