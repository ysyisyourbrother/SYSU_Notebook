nums db 5 ;文件数量
program1:
    name1 db "Perinfor" ;文件名字
        db 0 
    size1 db "170" ;文件大小单位字节
        db 0 
    sector1 db 8 ;扇区号
        db 0
program2:
    name2 db "stone" ;文件名字
        db 0 
    size2 db "403" ;文件大小单位字节
        db 0 
    sector2 db 9 ;扇区号
        db 0
program3:
    name3 db "square" ;文件名字
        db 0 
    size3 db "327" ;文件大小单位字节
        db 0 
    sector3 db 10 ;扇区号
        db 0

program4:
    name4 db "sandclock" ;文件名字
        db 0 
    size4 db "288" ;文件大小单位字节
        db 0 
    sector4 db 11 ;扇区号
        db 0

program5:
    name5 db "loading" ;文件名字
        db 0 
    size5 db "450" ;文件大小单位字节
        db 0 
    sector5 db 12 ;扇区号
        db 0
        
times 512 - ($ - $$) db 0 ;将前510字节不是0就填0
