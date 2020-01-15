    ; 定义常量
    R_t equ 1 ; 四个移动方向
    Lt_Dn_ equ 2
    Lt_Up_ equ 3
    delay equ 50000					; 计时器延迟计数,用于控制画框的速度
    ddelay equ 580					; 计时器延迟计数,用于控制画框的速度
org 8c00H

    ;使用int10h的清屏功能
    mov ah,6
    mov al,0
    mov ch,0  
    mov cl,0
    mov dh,24  
    mov dl,79
    mov bh,7 
    int 10h

start:
	xor ax,ax					; AX = 0
    mov ax,cs
	mov es,ax					; ES = 0
	mov ds,ax					; DS = CS
	mov es,ax					; ES = CS
	mov ax,0B800h				; 文本窗口显存起始地址
	mov gs,ax					; GS = B800h

loop1:
	dec word[count]	; 递减计数变量 产生延迟一共50000*580
	jnz loop1					; >0：跳转;
	mov word[count],delay
	dec word[dcount]				; 递减计数变量
    jnz loop1
	mov word[count],delay
	mov word[dcount],ddelay 

    ; 类似switch case语句
    mov al,1
    cmp al,byte[rdul]
	jz  Rt
    mov al,2
    cmp al,byte[rdul]
	jz  Lt_Dn
    mov al,3
    cmp al,byte[rdul]
	jz  Lt_Up
    jmp $	;rdul值异常 进入死循环


Rt:
	inc word[y]
	mov bx,word[y]
	mov ax,53
    sub ax,bx
    jz  RttoLt
    jmp show1

RttoLt:
    mov bx,word[x]
    mov ax,12
    sub ax,bx
    jz RttoLt_Dn
RttoLt_Up:
    mov word[y],52
    mov byte[rdul],Lt_Up_	
    jmp show1
RttoLt_Dn:
    mov word[y],52
    mov byte[rdul],Lt_Dn_
    jmp show1


Lt_Dn:
	inc word[x]
    dec word[y]
	mov bx,word[x]
    mov ax,25
    sub ax,bx
    jz  Lt_DntoRt
    jmp show1
Lt_DntoRt:
    mov word[x],24
    mov word[y],40
    mov byte[rdul],R_t	
    jmp show1


Lt_Up:
    dec word[x]
    dec word[y]
    mov bx,word[x]
    mov ax,11
	sub ax,bx
    jz  Lt_uptoRt
    jmp show1

Lt_uptoRt:
    mov word[x],12
    mov word[y],40
    mov byte[rdul],R_t
    jmp show1


show1:	
    mov ax,word[x]
    mov bx,80
    mul bx		;把 al*bx结果 放到 ax
    add ax,word[y]
    mov bx,2
    mul bx
    mov bp,ax
    mov ah, [property] ; 设置属性
    mov al,byte[char]			;  AL = 显示字符值（默认值为20h=空格符）
    mov word[gs:bp],ax  		;  显示字符的ASCII码值

    add byte[property],15

    ; 利用int 16h的 01号功能，输入空格后弹出程序
    mov ah,1
    int 16h
    mov bl,20h
    cmp al,bl
    jz Quit
    jmp loop1

Quit:
    ret
end:
    jmp $                   ; 停止画框，无限循环 
	
datadef:	
    count dw delay 
    dcount dw ddelay
    rdul db R_t      ; 向右运动

    x dw 12;    小球起始位置
    y dw 52
    property db 2
    char db '0'
times 512 - ($ - $$) db 0 ;将前510字节不是0就填0