org  7c00h		; BIOS将把引导扇区加载到0:7C00h处，并开始执行
OS_offset equ 0A100h  ;必须是100的后缀，因为.com文件中有org 0100所以要对上
;此程序是用来加载操作系统内核进入内存兵跳转运行！

ReadOs:
    mov ax,cs                ;段地址 ; 存放数据的内存基地址
    mov es,ax                ;设置段地址（不能直接mov es,段地址）
    mov bx, OS_offset   ;存放内核的内存偏移地址OS_offset
    mov ah,2                 ;功能号
    mov al,6                 ;扇区数，内核占用扇区数  注意：不止加载了一个扇区
    mov dl,0                 ;驱动器号 ; 软盘为0，硬盘和U盘为80H
    mov dh,0                 ;磁头号 ; 起始编号为0
    mov ch,0                 ;柱面号 ; 起始编号为0
    mov cl,3                 ;存放内核的起始扇区号 ; 起始编号为1
    int 13H 					;调用读磁盘BIOS的13h功能
    jmp 0A00h:100h			;控制权移交给内核

times 510 - ($ - $$) db 0 ;将前510字节不是0就填0
db 0x55
db 0xaa