    delay equ 50000					; 计时器延迟计数,用于控制画框的速度
    ddelay equ 1580					; 计时器延迟计数,用于控制画框的速度

org 8c00H
start:
	xor ax,ax	                ; 让ES DS CS都为0				
    mov ax,cs
	mov es,ax					
	mov ds,ax					
	mov es,ax					
	mov ax,0B800h				; 文本窗口显存起始地址
	mov gs,ax					; GS = B800h

    mov dh,5; 初始位置

loop1:
    dec word[count]	; 递减计数变量 产生延迟一共50000*580
	jnz loop1					; >0：跳转;
	mov word[count],delay
	dec word[dcount]				; 递减计数变量
    jnz loop1
	mov word[count],delay
	mov word[dcount],ddelay

    ;使用int10h的清屏功能 并保护dx
    push dx
    mov ah,6
    mov al,0
    mov ch,0;1  
    mov cl,0
    mov dh,24;12
    mov dl,79;38
    mov bh,7 
    int 10h
    pop dx

showMessage:
    inc dh
    mov al,11
    cmp dh,al
    jnz next
    mov dh,0
next:
    mov ax,Name
    mov bp,ax ; es:bp串地址
    mov cx,20
    mov ax,1301h ; ah=13 显示字符串
    mov bx,000ah ;bl 6-4背景色  3位1前景色高亮  2-0 前景色RGB
    mov dl,0Eh
    int 10h

    inc dh
    mov ax,Number
    mov bp,ax
    mov cx,15
    mov ax,1301h
    mov bx,000ch
    mov dl,0Eh
    int 10h
    sub dh,1

    ; 利用int 16h的 01号功能，输入空格后弹出程序
    mov ah,1
    int 16h
    mov bl,20h
    cmp al,bl
    jz Quit
    jmp loop1

Quit:
    ret

Name: db "Author:Ye Sheng Yuan"
Number: db "Number:17341190"
count dw delay 
dcount dw ddelay
times 512 - ($ - $$) db 0 ;将前510字节不是0就填0