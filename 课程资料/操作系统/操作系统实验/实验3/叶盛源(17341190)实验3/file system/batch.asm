batch_processing:
    instru db "-p1 -p5 -h -qb",'\0' ;执行一串指令

times 512 - ($ - $$) db 0 ;将前510字节不是0就填0