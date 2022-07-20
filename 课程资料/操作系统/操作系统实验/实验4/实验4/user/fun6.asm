org 0DC00H
Program:
	int 33
	int 34
	int 35
	int 36

Quit:
    ret

times 512 - ($ - $$) db 0 ;将前510字节不是0就填0