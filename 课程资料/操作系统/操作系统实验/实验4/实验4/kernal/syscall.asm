;实现BIOS基础的系统调用

; 导入全局变量
extern _in
extern _str

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

public _cls
_cls proc 
; 清屏
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
	ret
_cls endp

public _Readchar
_Readchar proc
;输入单个到字符到al
	mov ah,0
    int 16h
	mov byte ptr [_in],al
	ret
_Readchar endp

;输出字符
public _printChar
_printChar proc 
	push bp
	mov bp,sp
	mov al,[bp+4];char\ip\bp  bp 才能寻址
	mov bl,0
	mov ah,0eh
	int 10h
	mov sp,bp
	pop bp
	ret
_printChar endp

OffSetOfUserPrg equ 8c00h

public _RunProm
_RunProm proc
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
	mov cl,[bp+10]            ;起始扇区号 ; 起始P编号为1
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
	ret  ;中断用iret
_RunProm endp

OffSetOfUser6 equ 0DC00h
public _loadUser6FromDisk
_loadUser6FromDisk proc
	push ds	
	push es
	mov ax,cs                ;段地址 ; 存放数据的内存基地址
	mov es,ax                ;设置段地址（不能直接mov es,段地址）
	mov bx, OffSetOfUser6  ;偏移地址; 存放数据的内存偏移地址
	mov ah,2                 ; 功能号
	mov al,1                 ;扇区数
	mov dl,0                 ;驱动器号 ; 软盘为0，硬盘和U盘为80H
	mov dh,1                 ;磁头号 ; 起始编号为0  从19扇区开始
	mov ch,0                 ;柱面号 ; 起始编号为0
	mov cl,6            ;起始扇区号 
	int 13H ;                调用读磁盘BIOS的13h功能
	call bx ;执行用户程序
	pop  es
	pop  ds  
_loadUser6FromDisk endp

OffSetOfList equ 9c00h
public _loadListFromDisk
_loadListFromDisk proc
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
	ret
_loadListFromDisk endp

OffSetOfbatch equ 0Bc00h
public _loadbatchFromDisk
_loadbatchFromDisk proc
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
	ret
_loadbatchFromDisk endp

public _readbatch
_readbatch proc
	push ax
	mov ax,0Bc00h
	mov word ptr [_str],ax
	pop ax
	ret
_readbatch endp

public _readList
_readList proc
	push bp
	mov bp,sp
	mov al,[bp+4]
	add ax,9c00h
	mov word ptr [_str],ax
	pop bp
	ret
_readList endp

;int_20h时钟中断
Timer:
;保护现场
    push ax		
	push bx
	push cx
	push dx
	push bp
    push es
	push ds

;执行五次中断才运行一次风火轮，防止运行速度过快
    dec byte ptr es:[ccount]
	jz  DUMP 
	jmp end1
DUMP:
	mov byte ptr es:[ccount],ddelay
	inc byte ptr es:[state]
    cmp byte ptr es:[state],1
	jz  state1
	cmp byte ptr es:[state],2
	jz  state2
	cmp byte ptr es:[state],3
	jz  state3
	cmp byte ptr es:[state],4
	jz  state4

state1: 
    mov bp,offset char1
	jmp show_state
state2:
    mov bp,offset char2
	jmp show_state
state3:
    mov bp,offset char3
	jmp show_state
state4:
    mov bp,offset char4
	mov byte ptr es:[state],0  ;当到达第四个状态之后将state归0
	jmp show_state
show_state:
    mov ah,13h 	                    
	mov al,0                     	
	mov bl,0Fh 	                      
	mov bh,0 	                    
	mov dh,24 	                      
	mov dl,78 	                    
	mov cx,1 	                     
	int 10h 	                    	

	mov byte ptr es:[ccount],ddelay		;显示了一次字符后将ccount计数器重置	

end1: 
    mov al,20h
	out 20h,al			; 发送EOI到主8529A 让中断停止
	out 0A0h,al

;恢复现场
	pop ds
	pop es 
	pop bp
	pop dx 
	pop cx
	pop bx
	pop ax
	iret ;中断要用iret

    state db 0
    char1 db '|'
	char2 db '/'
	char3 db '-'
	char4 db '\'
	ddelay equ 5					        ; 计时器延迟时钟中断次数
	ccount db ddelay					     ; 计时器计数变量，初值=ddelay	

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

;用户自定义中断：
int_33:
	mov ax,1
	push ax
	call _RunProm
	pop ax
	iret
int_34:
	mov ax,2
	push ax
	call _RunProm
	pop ax
	iret
int_35:
	mov ax,3
	push ax
	call _RunProm
	pop ax
	iret
int_36:
	mov ax,4
	push ax
	call _RunProm
	pop ax
	iret

Data:
	int_09_saved dw 0,0