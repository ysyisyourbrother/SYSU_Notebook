    delay equ 50000					; 计时器延迟计数,用于控制画框的速度
    ddelay equ 2580					; 计时器延迟计数,用于控制画框的速度
org 100H
start:
	xor ax,ax					; AX = 0   程序加载到0000：100h才能正确执行
    mov ax,cs					; 开机这些寄存器都为0
	mov es,ax					; ES = 0
	mov ds,ax					; DS = CS
	mov es,ax					; ES = CS
	mov	ax,0B800h				; 文本窗口显存起始地址
	mov	gs,ax					; GS = B800h

	mov ah,6					;清屏成蓝色 显示在显示屏幕中部
    mov al,0 ;0全屏幕为空白
    mov ch,6 ;左上角行号
	mov cl,20 ;左上角列号
	mov dh,18;右下角行号
	mov dl,58;右下角列号
	mov bh,12h;卷入行属性 
	int 10h
	
showchar:
	mov cx,9					;显示几行
	mov ax,str0
	mov si,ax
	mov bp,1332				;显示图案的第一个位置 x*160+y*2
row:
	push cx
	mov cx,27					;一行的字符数
	
loop2:	
	mov ah,5eh					;设置显示字符的属性值
	mov al,[si]			;  AL = 显示字符值（默认值为20h=空格符）
	mov word[gs:bp],ax
	inc si
	add bp,2
	sub cx,1
	jnz loop2
	add bp,106		;(80-字符数)*2 换行
	pop cx
	sub cx,1   ;修改Flag寄存器
	jnz row
	
	mov cx,25			; 进度条长度	
loop1:
	;延时  dos int 15h功能调用  cx 和 dx的值大小影响延时的时间
	push dx					
	push cx
	push ax
	mov ah,86h
    mov cx,02h
    mov dx,0x9999     
    int 15h
	pop ax
	pop cx
	pop dx
	
	mov dx,cx
	test dx,01h			;偶数跳show2
	jz show2

	
show1:					;loading..
	push cx
	mov al,1
	mov ah,13h
	mov bl,1fh
	mov bh,0
	mov dh,17			;行
	mov dl,36			;列
	mov cx,11			
	mov bp,str1
	int 10h
	pop cx
	jmp show
	
show2:
	push cx
	mov al,1
	mov ah,13h
	mov bl,1fh
	mov bh,0
	mov dh,17  ;行
	mov dl,36  ;列
	mov cx,11
	mov bp,str2
	int 10h
	pop cx
	jmp show

	
show:
	; 展示进度条
    mov ax,word[x]
	mov bx,80
	mul bx
	add ax,word[y]
	mov bx,2
	mul bx
	mov bp,ax
	mov ah,20h					;  
	mov al,20h			;  AL = 显示字符值（默认值为20h=空格符）
	mov word[gs:bp],ax  		;  显示字符的ASCII码值
	inc word[y]
	dec cx
	jnz loop1
	
return0: 
	jmp $
	


datadef:	
    x    dw 18
    y    dw 27			
   str0 db  "     welcome to my OS!     "
		db  "                           "
		db  "       author: yeshy       "
		db  "                           "
		db  "     **    *    *****      "
		db  "     * *   *    *    *     "
		db  "     *  *  *    *****      "
		db  "     *   * *    *    *     "
		db  "     *    **    *****      "
	str1 db  "LOADING..  "
	str2 db  "LOADING...."
times 512 - ($ - $$) db 0 ;将前510字节不是0就填0

