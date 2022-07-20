;   NASM汇编格式
    D_n equ 1
    U_p equ 2
    L_t equ 3
    R_t equ 4
    delay equ 50000					; 计时器延迟计数,用于控制画框的速度
    ddelay equ 580					; 计时器延迟计数,用于控制画框的速度


    ;使用int10h的清屏功能
    mov ah,6
    mov al,0
    mov ch,0  
    mov cl,0
    mov dh,24  
    mov dl,79
    mov bh,7 
    int 10h

showMessage:
    mov ax,Message
    mov bp,ax ; es:bp串地址
    mov cx,11
    mov ax,1301h ; ah=13 显示字符串
    mov bx,101ah ;bl 6-4背景色  3位1前景色高亮  2-0 前景色RGB
    mov dx,0537h
    int 10h

start:
	xor ax,ax					; AX = 0   程序加载到0000：7c00h才能正确执行(bios的引导扇区地址)
    mov ax,cs
	mov es,ax					; ES = 0
	mov ds,ax					; DS = CS
	mov es,ax					; ES = CS
	mov ax,0B800h				; 文本窗口显存起始地址
	mov gs,ax					; GS = B800h

loop1:
	dec word[count]	;count等于delay,dcount等于ddelay ; 递减计数变量
	jnz loop1					; >0：跳转;
	mov word[count],delay
	dec word[dcount]				; 递减计数变量
    jnz loop1
	mov word[count],delay
	mov word[dcount],ddelay ;延迟50000*580次

    mov al,1
    cmp al,byte[rdul]
	jz  Dn
    mov al,2
    cmp al,byte[rdul]
	jz  Up
    mov al,3
    cmp al,byte[rdul]
	jz  Lt
    mov al,4
    cmp al,byte[rdul]
	jz  Rt
    jmp $	
	
Dn:
    inc word[x2]
    mov bx,word[x2]
    mov ax,12
    sub ax,bx
    jz DntoRt
    add byte[property],15
    jmp show
DntoRt:
    mov word[x2],11
    mov byte[rdul],R_t
    jmp show

Rt:
    inc word[y2]
    mov bx,word[y2]
    mov ax,80
    sub ax,bx
    jz RttoUp
    add byte[property],15
    jmp show
RttoUp:
    mov word[y2],79
    mov byte[rdul],U_p
    jmp show

Up:
    dec word[x2]
    mov ax,-1
    mov bx,word[x2]
    sub ax,bx
    jz UptoLt
    add byte[property],15
    jmp show
UptoLt:
    mov word[x2],0
    mov byte[rdul],L_t
    jmp show

Lt:
    dec word[y2]
    mov bx,word[y2]
    mov ax,39
    sub ax,bx
    jz LttoDn
    add byte[property],15
    jmp show
LttoDn:
    mov word[y2],40
    mov byte[rdul],D_n
    jmp show

show:
    xor ax,ax                 ; 计算显存地址
    mov ax,word[x2]
    mov bx,80
    mul bx		;al*bx 放到 ax
    add ax,word[y2]
    mov bx,2
    mul bx
    mov bp,ax
    mov ah,[property]
    mov al,byte[char]			;  AL = 显示字符值（默认值为20h=空格符）
    mov word[gs:bp],ax  		;  显示字符的ASCII码值
    
	
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
    rdul db D_n

    Message: db "Hello World!"
    x2 dw 0
    y2 dw 40
    char db 'O'
    property db 1