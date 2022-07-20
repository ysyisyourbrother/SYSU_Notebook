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
	push ds	
	push es
	push bp
	mov bp,sp
	mov ax,cs                ;段地址 ; 存放数据的内存基地址
	mov es,ax                ;设置段地址（不能直接mov es,段地址）
	mov bx, OffSetOfUserPrg  ;偏移地址; 存放数据的内存偏移地址
	mov ah,2                 ; 功能号
	mov al,1                 ;扇区数
	mov dl,0                 ;驱动器号 ; 软盘为0，硬盘和U盘为80H
	mov dh,0                 ;磁头号 ; 起始编号为0
	mov ch,0                 ;柱面号 ; 起始编号为0
	mov cl,[bp+8]            ;起始扇区号 ; 起始编号为1
	int 13H ;                调用读磁盘BIOS的13h功能
	; 用户程序a.com已加载到指定内存区域
	mov bx,OffSetOfUserPrg 
	call bx ;执行用户程序
	pop bp
	pop  es
	pop  ds  
	ret
_RunProm endp

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
	mov dh,0                 ;磁头号 ; 起始编号为0
	mov ch,0                 ;柱面号 ; 起始编号为0
	mov cl,2            ;起始扇区号  起始编号为1  list在第二个扇区
	int 13H ;                调用读磁盘BIOS的13h功能
	; 用户程序a.com已加载到指定内存区域
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
	mov dh,0                 ;磁头号 ; 起始编号为0
	mov ch,0                 ;柱面号 ; 起始编号为0
	mov cl,13            ;起始扇区号  起始编号为1  默认batch在第十三个扇区
	int 13H ;                调用读磁盘BIOS的13h功能
	; 用户程序a.com已加载到指定内存区域
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
_readList endp



