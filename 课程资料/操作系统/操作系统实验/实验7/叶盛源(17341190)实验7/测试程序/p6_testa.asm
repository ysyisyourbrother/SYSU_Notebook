extrn _main:near

.8086
_TEXT segment byte public 'CODE'
DGROUP group _TEXT,_DATA,_BSS
       assume cs:_TEXT

org 100h

start:
    mov ax, cs
    mov ds, ax
    mov es, ax
    mov ss, ax
    call near ptr _main
    jmp $

    include process.asm
    
_TEXT ends
_DATA segment word public 'DATA'
_DATA ends
_BSS segment word public 'BSS'
_BSS ends
end start