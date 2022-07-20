;实现BIOS基础的系统调用

; 导入全局变量
extern _in
extern _str
extern _Segment
extern _sector_number

extrn _Current_Process
extrn _Save_Process
extrn _Schedule
extrn _Have_Program
extrn _special
extrn _Program_Num
extrn _CurrentPCBno
extrn _Segment

;****************************
; SCOPY@                    *
;****************************
;局部字符串带初始化作为实参问题补钉程序
public SCOPY@
SCOPY@ proc 
	arg_0 = dword ptr 6
	arg_4 = dword ptr 0ah
	push bp
	mov bp,sp
	push si
	push di
	push ds
	lds si,[bp+arg_0]
	les di,[bp+arg_4]
	cld
	shr cx,1
	rep movsw
	adc cx,cx
	rep movsb
	pop ds
	pop di
	pop si
	pop bp
	retf 8
SCOPY@ endp

;输出字符
public _printChar
_printChar proc 
	mov ah,0
	int 21h
	ret
_printChar endp

;输入单个到字符到al
public _Readchar
_Readchar proc
	mov ah,1
    int 21h
	mov byte ptr [_in],al
	ret
_Readchar endp

public _cls
_cls proc 
	mov ah,2
	int 21h
	ret
_cls endp

OffSetOfUserPrg equ 8c00h

public _RunProm
_RunProm proc
	mov ah,3
	int 21h
	ret  ;中断用iret
_RunProm endp

OffSetOfList equ 9c00h
public _loadListFromDisk
_loadListFromDisk proc
	mov ah,5
	int 21h 
	ret
_loadListFromDisk endp

OffSetOfbatch equ 0Bc00h
public _loadbatchFromDisk
_loadbatchFromDisk proc
	mov ah,6
	int 21h
	ret
_loadbatchFromDisk endp

public _readbatch
_readbatch proc
	mov ah,7
	int 21h
	ret
_readbatch endp

public _readList
_readList proc
	mov ah,8
	int 21h
	ret
	; push bp
	; mov bp,sp
	; mov al,[bp+4]
	; add ax,9c00h
	; mov word ptr [_str],ax
	; pop bp
	; ret
_readList endp

;=========================================================================
;					void _SetTimer()
;=========================================================================
public _SetTimer
_SetTimer proc
    push ax
    mov al,34h   ; 设控制字值 
    out 43h,al   ; 写控制字到控制字寄存器 
    mov ax,29830 ; 每秒 20 次中断（50ms 一次） 
    out 40h,al   ; 写计数器 0 的低字节 
    mov al,ah    ; AL=AH 
    out 40h,al   ; 写计数器 0 的高字节 
	pop ax
	ret
_SetTimer endp

public _setClock
_setClock proc
    push ax
	push bx
	push cx
	push dx
	push ds
	push es
	
    call near ptr _SetTimer
    xor ax,ax
	mov es,ax
	mov word ptr es:[20h],offset Pro_Timer ;改时间中断
	mov ax,cs
	mov word ptr es:[22h],cs
	
	pop ax
	mov es,ax
	pop ax
	mov ds,ax
	pop dx
	pop cx
	pop bx
	pop ax
	ret
_setClock endp


;=========================================================================
;					void _run_process(int start, int seg)
;=========================================================================
public _another_RunProm
_another_RunProm proc
    push ax
	push bp
	
	mov bp,sp
	
    mov ax,[bp+6]      	;段地址 ; 存放数据的内存基地址
	mov es,ax          	;设置段地址（不能直接mov es,段地址）
	mov bx,100h        	;偏移地址; 存放数据的内存偏移地址
	mov ah,2           	;功能号
	mov al,1          	;扇区数
	mov dl,0          	;驱动器号 ; 软盘为0，硬盘和U盘为80H
	mov dh,1          	;磁头号 ; 起始编号为0
	mov ch,0          	;柱面号 ; 起始编号为0
	mov cl,[bp+8]       ;起始扇区号 ; 起始编号为1
	int 13H          	; 调用中断
	
	pop bp
	pop ax
	
	ret
_another_RunProm endp
;===========================================================================================
;                                    中断服务程序
;===========================================================================================

;================================================ 
;              int_20用户程序时钟中断
;================================================
Finite dw 0	
Pro_Timer: ; 用户程序执行的时候的20h号中断程序
;*****************************************
;*                Save                   *
; ****************************************
    cmp word ptr[_Program_Num],0
	
	jnz Save
	jmp No_Progress
Save:
	inc word ptr[Finite]
	cmp word ptr[Finite],400  ;总调度次数
	jnz Lee
    mov word ptr[_CurrentPCBno],0
	mov word ptr[Finite],0
	mov word ptr[_Program_Num],0
	mov word ptr[_Segment],2000h
	jmp Pre
Lee:
    push ss
	push ax
	push bx
	push cx
	push dx
	push sp
	push bp
	push si
	push di
	push ds
	push es
	.386
	push fs
	push gs
	.8086

	mov ax,cs
	mov ds, ax
	mov es, ax

	call near ptr _Save_Process
	call near ptr _Schedule 
	
Pre:
	mov ax, cs
	mov ds, ax
	mov es, ax
	
	call near ptr _Current_Process   ; 获取要调度的进程的PCB块，返回值放在了ax里面
	mov bp, ax

	mov ss,word ptr ds:[bp+0]; 在PCB块内寻找ss和sp的地址
	mov sp,word ptr ds:[bp+16] 

	cmp word ptr ds:[bp+32],0  ;查看当前状态是不是new
	jnz No_First_Time ;如果是new状态说明是第一次

;*****************************************
;*                Restart                *
; ****************************************
Restart:
    call near ptr _special

	; 没有push ss 和 sp的值因为已经赋值了
	push word ptr ds:[bp+30]
	push word ptr ds:[bp+28]
	push word ptr ds:[bp+26]
	
	push word ptr ds:[bp+2]
	push word ptr ds:[bp+4]
	push word ptr ds:[bp+6]
	push word ptr ds:[bp+8]
	push word ptr ds:[bp+10]
	push word ptr ds:[bp+12]
	push word ptr ds:[bp+14]
	push word ptr ds:[bp+18]
	push word ptr ds:[bp+20]
	push word ptr ds:[bp+22]
	push word ptr ds:[bp+24]

	pop ax
	pop cx
	pop dx
	pop bx
	pop bp
	pop si
	pop di
	pop ds
	pop es
	.386
	pop fs
	pop gs
	.8086

	; 发送AEIO 结束中断
	push ax         
	mov al,20h
	out 20h,al
	out 0A0h,al
	pop ax
	iret

No_First_Time:	
	add sp,16 
	jmp Restart
	
No_Progress:
    call another_Timer
	
	push ax         
	mov al,20h
	out 20h,al
	out 0A0h,al
	pop ax
	iret

another_Timer:
    push ax
	push bx
	push cx
	push dx
	push bp
    push es
	push ds

	
	mov ax,cs
	mov ds,ax

	dec byte ptr [ds:ccount]
	jnz end1

	cmp byte ptr [ds:count],0
	jz case1
	cmp byte ptr [ds:count],1
	jz case2
	cmp byte ptr [ds:count],2
	jz case3
	
case1:	
    inc byte ptr [ds:count]
	mov al,'/'
	jmp show
case2:	
    inc byte ptr [ds:count]
	mov al,'\'
	jmp show
case3:	
    mov byte ptr [ds:count],0
	mov al,'|'
	jmp show
	
show:
    mov bx,0b800h
	mov es,bx
	mov ah,0ah
	mov es:[((80*24+78)*2)],ax
	
	mov byte ptr [ds:ccount],5

end1:
	pop ax
	mov ds,ax
	pop ax
	mov es,ax
	pop bp
	pop dx
	pop cx
	pop bx
	pop ax

	ret

	count db 0
	ccount db 5

;键盘中断
;要在进入用户程序之后改变int 9的cs ip
int_09:
;保护现场
    push ax								
	push bx
	push cx
	push dx
	push bp
    push es
	push ds

    mov ah,13h 	                        ; 功能号
	mov al,0                     		; 光标返回起始位置
	mov bl,0Fh 	                        ; 0000：黑底、1111：亮白字
	mov bh,0 	                    	; 
	mov dh,15 	                        ; 行
	mov dl,45 	                        ; 列
	mov cx,11 	                        ; 串长
	mov bp,offset me
	int 10h

    ;延时  dos int 15h功能调用  cx 和 dx的值大小影响延时的时间
	push dx					
	push cx
	push ax
	mov ah,86h
    mov cx,02h
    mov dx,9999h     
    int 15h
	pop ax
	pop cx
	pop dx

	;清除该ouch位置的字符串
	mov ah,6
    mov al,0
	mov ch,15
	mov cl,45
	mov dh,15
	mov dl,55
	mov bh,7
	int 10H
    
    in al,60h
	mov al,20h					    ; AL = EOI
	out 20h,al						; 发送EOI到主8529A
	out 0A0h,al					    ; 发送EOI到从8529A

	
;还原现场
	pop ds
	pop es
	pop bp
	pop dx
	pop cx
	pop bx
	pop ax
	
	iret

me db "OUCH! OUCH!"

keyboard_vector dw 0

;;;;;;;;;;;;;;;;;;;;;;;;;;;
;         int21        
;     在ah设置功能号调用  
;     完成系统内核的调用       
;;;;;;;;;;;;;;;;;;;;;;;;;;;    
int_21h:
	cmp ah,0
	jz printChar
	cmp ah,1
	jnz next1
	jmp Readchar
next1:
	cmp ah,2
	jnz next2
	jmp clear
next2:
	cmp ah,3
	jnz next3
	jmp RunProm
next3:
	
next4:
	cmp ah,5
	jnz next5
	jmp loadListFromDisk
next5:
	cmp ah,6
	jnz next6
	jmp loadbatchFromDisk
next6:
	cmp ah,7
	jnz next7
	jmp readbatch
next7:
	cmp ah,8
	jnz quit
	jmp readList
quit:
	iret
	jmp $

printChar:
	push bp
	mov bp,sp
	mov al,[bp+10];char\ip\bp\ip\cs\psw  bp 才能寻址  
	mov bl,0
	mov ah,0eh
	int 10h
	mov sp,bp
	pop bp
	iret
Readchar:;输入单个到字符到al
	mov ah,0
    int 16h
	mov byte ptr [_in],al
	iret
clear:
	push ax
	push bx
	push cx
	push dx		
	mov	ax, 600h	; AH = 6,  AL = 0
	mov	bx, 700h	; 黑底白字(BL = 7)
	mov	cx, 0		; 左上角: (0, 0)
	mov	dx, 184fh	; 右下角: (24, 79)
	int	10h		; 显示中断

	mov ah,2
	mov bh,0
	mov dx,0
	int 10h

	pop dx
	pop cx
	pop bx
	pop ax
	iret

RunProm:
	push ax
	push ds	
	push es
	push bp
	
	;保护键盘中断
	xor ax, ax
	mov es, ax
	mov bp,offset int_09_saved
	mov word ptr ax,es:[36]
	mov word ptr [bp],ax
	mov word ptr ax,es:[38]
	mov word ptr [bp+2],ax 
	;修改键盘中断
	mov word ptr es:[9*4], offset int_09
	mov word ptr es:[9*4+2], cs

	mov bp,sp
	mov ax,cs                ;段地址 ; 存放数据的内存基地址
	mov es,ax                ;设置段地址（不能直接mov es,段地址）
	mov bx, OffSetOfUserPrg  ;偏移地址; 存放数据的内存偏移地址
	mov ah,2                 ; 功能号
	mov al,1                 ;扇区数
	mov dl,0                 ;驱动器号 ; 软盘为0，硬盘和U盘为80H
	mov dh,1                 ;磁头号 ; 起始编号为0
	mov ch,0                 ;柱面号 ; 起始编号为0
	mov cl,[bp+16]            ;起始扇区号 ; 起始P编号为1
	int 13H ;                调用读磁盘BIOS的13h功能
	;用户程序a.com已加载到指定内存区域
	call bx ;执行用户程序
	

	;恢复键盘中断
	xor ax, ax
	mov es, ax
	mov bp,offset int_09_saved
	mov word ptr ax,[bp]
	mov word ptr es:[36],ax
	mov word ptr ax,[bp+2]
	mov word ptr es:[38],ax

	pop  bp
	pop  es
	pop  ds
	pop  ax
	iret

loadListFromDisk:
	push ds	
	push es
	mov ax,cs                ;段地址 ; 存放数据的内存基地址
	mov es,ax                ;设置段地址（不能直接mov es,段地址）
	mov bx, OffSetOfList  ;偏移地址; 存放数据的内存偏移地址
	mov ah,2                 ; 功能号
	mov al,1                 ;扇区数
	mov dl,0                 ;驱动器号 ; 软盘为0，硬盘和U盘为80H
	mov dh,0                 ;磁头号 ; 起始编号为0  从19扇区开始
	mov ch,0                 ;柱面号 ; 起始编号为0
	mov cl,2            ;起始扇区号  起始编号为1  list在第二个扇区
	int 13H ;                调用读磁盘BIOS的13h功能
	pop  es
	pop  ds  
	iret

loadbatchFromDisk:
	push ds	
	push es
	mov ax,cs                ;段地址 ; 存放数据的内存基地址
	mov es,ax                ;设置段地址（不能直接mov es,段地址）
	mov bx, OffSetOfbatch  ;偏移地址; 存放数据的内存偏移地址
	mov ah,2                 ; 功能号
	mov al,1                 ;扇区数
	mov dl,0                 ;驱动器号 ; 软盘为0，硬盘和U盘为80H
	mov dh,1                 ;磁头号 ; 起始编号为0
	mov ch,0                 ;柱面号 ; 起始编号为0
	mov cl,7            ;起始扇区号  起始编号为1  默认batch在第1磁头 第7个扇区
	int 13H ;                调用读磁盘BIOS的13h功能
	pop  es
	pop  ds  
	iret

readbatch:
	push ax
	mov ax,0Bc00h
	mov word ptr [_str],ax
	pop ax
	iret

readList:
	push ax
	push bp
	mov bp,sp
	xor ax,ax  ;  要先将ax清零！！！
	mov al,[bp+12]
	add ax,9c00h
	mov word ptr [_str],ax
	pop bp
	pop ax
	iret

Data:
	int_09_saved dw 0,0