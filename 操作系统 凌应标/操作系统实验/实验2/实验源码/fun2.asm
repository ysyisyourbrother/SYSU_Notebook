    ; 定义常量
    Dn_Rt equ 1 ; 四个移动方向
    Up_Rt equ 2
    Up_Lt equ 3
    Dn_Lt equ 4
    delay equ 50000					; 计时器延迟计数,用于控制画框的速度
    ddelay equ 580					; 计时器延迟计数,用于控制画框的速度


    ;使用int10h的清屏功能 并保护dx
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
	jz  DnRt
    mov al,2
    cmp al,byte[rdul]
	jz  UpRt
    mov al,3
    cmp al,byte[rdul]
	jz  UpLt
    mov al,4
    cmp al,byte[rdul]
	jz  DnLt
    jmp $	;rdul值异常 进入死循环


DnRt:
	inc word[x]
	inc word[y]
	mov bx,word[x]
	mov ax,25 ;共25行
    sub ax,bx
    jz  dr2ur
    mov bx,word[y]
	mov ax,40 ;每行80个字符
	sub ax,bx
    jz  dr2dl
    jmp show1
dr2ur:
    mov word[x],23
    mov byte[rdul],Up_Rt	
    jmp show1
dr2dl:
    mov word[y],38
    mov byte[rdul],Dn_Lt	
    jmp show1

UpRt:
	dec word[x]
    inc word[y]
    mov bx,word[y]
	mov ax,40
	sub ax,bx
    jz  ur2ul
	mov bx,word[x]
    mov ax,10
    sub ax,bx
    jz  ur2dr
    jmp show1
ur2ul:
    mov word[y],38
    mov byte[rdul],Up_Lt	
    jmp show1
ur2dr:
    mov word[x],12
    mov byte[rdul],Dn_Rt	
    jmp show1

UpLt:
    dec word[x]
    dec word[y]
    mov bx,word[x]
    mov ax,10
	sub ax,bx
    jz  ul2dl
    mov bx,word[y]
    mov ax,-1
    sub ax,bx
    jz  ul2ur
    jmp show1

ul2dl:
    mov word[x],12
    mov byte[rdul],Dn_Lt	
    jmp show1
ul2ur:
    mov word[y],1
    mov byte[rdul],Up_Rt	
    jmp show1

DnLt:
    inc word[x]
    dec word[y]
    mov bx,word[y]
    mov ax,-1
    sub ax,bx
    jz  dl2dr
    mov bx,word[x]
    mov ax,25
    sub ax,bx
    jz  dl2ul
    jmp show1

dl2dr:
    mov word[y],1
    mov byte[rdul],Dn_Rt	
    jmp show1
	
dl2ul:
    mov word[x],23
    mov byte[rdul],Up_Lt	
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
    jmp ret    
end:
    jmp $                   ; 停止画框，无限循环 
	
datadef:	
    count dw delay 
    dcount dw ddelay
    rdul db Dn_Rt         ; 向右下运动

    x dw 15;    小球起始位置
    y dw 0
    property db 2
    char db '0'