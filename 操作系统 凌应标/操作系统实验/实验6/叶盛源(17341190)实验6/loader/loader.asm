org  7c00h		; BIOS将把引导扇区加载到0:7C00h处，并开始执行
;此程序是用来加载操作系统内核进入内存并跳转运行！

;读软盘或硬盘上的kernal到内存的ES:BX处：
ReadOs:
    mov ax, SegOfKernal  	;段地址 ; 存放数据的内存基地址
    mov es,ax
    mov bx, OffSetOfKernal   ;存放内核的内存偏移地址OS_offset
    mov ah,2                 ;功能号
    mov al,8                 ;扇区数，内核占用扇区数  注意：不止加载了一个扇区
    mov dl,0                 ;驱动器号 ; 软盘为0，硬盘和U盘为80H
    mov dh,0                 ;磁头号 ; 起始编号为0
    mov ch,0                 ;柱面号 ; 起始编号为0
    mov cl,3                 ;存放内核的起始扇区号 ; 起始编号为1
    int 13H 					;调用读磁盘BIOS的13h功能
    jmp SegOfKernal:OffSetOfKernal			;控制权移交给内核
    jmp $

times 510 - ($ - $$) db 0 ;将前510字节不是0就填0
OffSetOfKernal equ 100h
SegOfKernal equ 64*1024/16  ;第二个64k内存的段地址  0x1000
db 0x55
db 0xaa