//go:build amd64

#include "textflag.h"

// func BranchlessSearchInt64(data unsafe.Pointer, len int, target int64) int
TEXT ·BranchlessSearchInt64(SB), NOSPLIT, $0-32
	MOVQ data+0(FP), AX
	MOVQ len+8(FP), BX
	MOVQ target+16(FP), CX

	TESTQ BX, BX
	JLE   ret_zero

	XORQ R8, R8

loop:
	CMPQ BX, $1
	JLE  end_loop

	MOVQ BX, R9
	SHRQ $1, R9

	LEAQ -1(R8)(R9*1), R10
	MOVQ (AX)(R10*8), R11

	LEAQ    (R8)(R9*1), R12
	CMPQ    R11, CX
	CMOVQLT R12, R8

	SUBQ R9, BX
	JMP  loop

end_loop:
	TESTQ BX, BX
	JLE   done

	MOVQ    (AX)(R8*8), R11
	LEAQ    1(R8), R12
	CMPQ    R11, CX
	CMOVQLT R12, R8

done:
	MOVQ R8, ret+24(FP)
	RET

ret_zero:
	MOVQ $0, ret+24(FP)
	RET

// func BranchlessSearchUint64(data unsafe.Pointer, len int, target uint64) int
TEXT ·BranchlessSearchUint64(SB), NOSPLIT, $0-32
	MOVQ data+0(FP), AX
	MOVQ len+8(FP), BX
	MOVQ target+16(FP), CX

	TESTQ BX, BX
	JLE   ret_zero

	XORQ R8, R8

loop:
	CMPQ BX, $1
	JLE  end_loop

	MOVQ BX, R9
	SHRQ $1, R9

	LEAQ -1(R8)(R9*1), R10
	MOVQ (AX)(R10*8), R11

	LEAQ    (R8)(R9*1), R12
	CMPQ    R11, CX
	CMOVQCS R12, R8

	SUBQ R9, BX
	JMP  loop

end_loop:
	TESTQ BX, BX
	JLE   done

	MOVQ    (AX)(R8*8), R11
	LEAQ    1(R8), R12
	CMPQ    R11, CX
	CMOVQCS R12, R8

done:
	MOVQ R8, ret+24(FP)
	RET

ret_zero:
	MOVQ $0, ret+24(FP)
	RET

// func BranchlessSearchInt32(data unsafe.Pointer, len int, target int32) int
TEXT ·BranchlessSearchInt32(SB), NOSPLIT, $0-32
	MOVQ data+0(FP), AX
	MOVQ len+8(FP), BX
	MOVL target+16(FP), CX

	TESTQ BX, BX
	JLE   ret_zero

	XORQ R8, R8

loop:
	CMPQ BX, $1
	JLE  end_loop

	MOVQ BX, R9
	SHRQ $1, R9

	LEAQ -1(R8)(R9*1), R10
	MOVL (AX)(R10*4), R11

	LEAQ    (R8)(R9*1), R12
	CMPL    R11, CX
	CMOVQLT R12, R8

	SUBQ R9, BX
	JMP  loop

end_loop:
	TESTQ BX, BX
	JLE   done

	MOVL    (AX)(R8*4), R11
	LEAQ    1(R8), R12
	CMPL    R11, CX
	CMOVQLT R12, R8

done:
	MOVQ R8, ret+24(FP)
	RET

ret_zero:
	MOVQ $0, ret+24(FP)
	RET

// func BranchlessSearchUint32(data unsafe.Pointer, len int, target uint32) int
TEXT ·BranchlessSearchUint32(SB), NOSPLIT, $0-32
	MOVQ data+0(FP), AX
	MOVQ len+8(FP), BX
	MOVL target+16(FP), CX

	TESTQ BX, BX
	JLE   ret_zero

	XORQ R8, R8

loop:
	CMPQ BX, $1
	JLE  end_loop

	MOVQ BX, R9
	SHRQ $1, R9

	LEAQ -1(R8)(R9*1), R10
	MOVL (AX)(R10*4), R11

	LEAQ    (R8)(R9*1), R12
	CMPL    R11, CX
	CMOVQCS R12, R8

	SUBQ R9, BX
	JMP  loop

end_loop:
	TESTQ BX, BX
	JLE   done

	MOVL    (AX)(R8*4), R11
	LEAQ    1(R8), R12
	CMPL    R11, CX
	CMOVQCS R12, R8

done:
	MOVQ R8, ret+24(FP)
	RET

ret_zero:
	MOVQ $0, ret+24(FP)
	RET

// func BranchlessSearchInt16(data unsafe.Pointer, len int, target int16) int
TEXT ·BranchlessSearchInt16(SB), NOSPLIT, $0-32
	MOVQ data+0(FP), AX
	MOVQ len+8(FP), BX
	MOVW target+16(FP), CX

	TESTQ BX, BX
	JLE   ret_zero

	XORQ R8, R8

loop:
	CMPQ BX, $1
	JLE  end_loop

	MOVQ BX, R9
	SHRQ $1, R9

	LEAQ -1(R8)(R9*1), R10
	MOVW (AX)(R10*2), R11

	LEAQ    (R8)(R9*1), R12
	CMPW    R11, CX
	CMOVQLT R12, R8

	SUBQ R9, BX
	JMP  loop

end_loop:
	TESTQ BX, BX
	JLE   done

	MOVW    (AX)(R8*2), R11
	LEAQ    1(R8), R12
	CMPW    R11, CX
	CMOVQLT R12, R8

done:
	MOVQ R8, ret+24(FP)
	RET

ret_zero:
	MOVQ $0, ret+24(FP)
	RET

// func BranchlessSearchUint16(data unsafe.Pointer, len int, target uint16) int
TEXT ·BranchlessSearchUint16(SB), NOSPLIT, $0-32
	MOVQ data+0(FP), AX
	MOVQ len+8(FP), BX
	MOVW target+16(FP), CX

	TESTQ BX, BX
	JLE   ret_zero

	XORQ R8, R8

loop:
	CMPQ BX, $1
	JLE  end_loop

	MOVQ BX, R9
	SHRQ $1, R9

	LEAQ -1(R8)(R9*1), R10
	MOVW (AX)(R10*2), R11

	LEAQ    (R8)(R9*1), R12
	CMPW    R11, CX
	CMOVQCS R12, R8

	SUBQ R9, BX
	JMP  loop

end_loop:
	TESTQ BX, BX
	JLE   done

	MOVW    (AX)(R8*2), R11
	LEAQ    1(R8), R12
	CMPW    R11, CX
	CMOVQCS R12, R8

done:
	MOVQ R8, ret+24(FP)
	RET

ret_zero:
	MOVQ $0, ret+24(FP)
	RET

// func BranchlessSearchInt8(data unsafe.Pointer, len int, target int8) int
TEXT ·BranchlessSearchInt8(SB), NOSPLIT, $0-32
	MOVQ data+0(FP), AX
	MOVQ len+8(FP), BX
	MOVB target+16(FP), CX

	TESTQ BX, BX
	JLE   ret_zero

	XORQ R8, R8

loop:
	CMPQ BX, $1
	JLE  end_loop

	MOVQ BX, R9
	SHRQ $1, R9

	LEAQ -1(R8)(R9*1), R10
	MOVB (AX)(R10*1), R11

	LEAQ    (R8)(R9*1), R12
	CMPB    R11, CX
	CMOVQLT R12, R8

	SUBQ R9, BX
	JMP  loop

end_loop:
	TESTQ BX, BX
	JLE   done

	MOVB    (AX)(R8*1), R11
	LEAQ    1(R8), R12
	CMPB    R11, CX
	CMOVQLT R12, R8

done:
	MOVQ R8, ret+24(FP)
	RET

ret_zero:
	MOVQ $0, ret+24(FP)
	RET

// func BranchlessSearchUint8(data unsafe.Pointer, len int, target uint8) int
TEXT ·BranchlessSearchUint8(SB), NOSPLIT, $0-32
	MOVQ data+0(FP), AX
	MOVQ len+8(FP), BX
	MOVB target+16(FP), CX

	TESTQ BX, BX
	JLE   ret_zero

	XORQ R8, R8

loop:
	CMPQ BX, $1
	JLE  end_loop

	MOVQ BX, R9
	SHRQ $1, R9

	LEAQ -1(R8)(R9*1), R10
	MOVB (AX)(R10*1), R11

	LEAQ    (R8)(R9*1), R12
	CMPB    R11, CX
	CMOVQCS R12, R8

	SUBQ R9, BX
	JMP  loop

end_loop:
	TESTQ BX, BX
	JLE   done

	MOVB    (AX)(R8*1), R11
	LEAQ    1(R8), R12
	CMPB    R11, CX
	CMOVQCS R12, R8

done:
	MOVQ R8, ret+24(FP)
	RET

ret_zero:
	MOVQ $0, ret+24(FP)
	RET

// func BranchlessSearchFloat64(data unsafe.Pointer, len int, target float64) int
TEXT ·BranchlessSearchFloat64(SB), NOSPLIT, $0-32
	MOVQ  data+0(FP), AX
	MOVQ  len+8(FP), BX
	MOVSD target+16(FP), X0

	TESTQ BX, BX
	JLE   ret_zero

	XORQ R8, R8

loop:
	CMPQ BX, $1
	JLE  end_loop

	MOVQ BX, R9
	SHRQ $1, R9

	LEAQ  -1(R8)(R9*1), R10
	MOVSD (AX)(R10*8), X1

	LEAQ    (R8)(R9*1), R12
	UCOMISD X0, X1
	CMOVQCS R12, R8

	SUBQ R9, BX
	JMP  loop

end_loop:
	TESTQ BX, BX
	JLE   done

	MOVSD   (AX)(R8*8), X1
	LEAQ    1(R8), R12
	UCOMISD X0, X1
	CMOVQCS R12, R8

done:
	MOVQ R8, ret+24(FP)
	RET

ret_zero:
	MOVQ $0, ret+24(FP)
	RET

// func BranchlessSearchFloat32(data unsafe.Pointer, len int, target float32) int
TEXT ·BranchlessSearchFloat32(SB), NOSPLIT, $0-32
	MOVQ  data+0(FP), AX
	MOVQ  len+8(FP), BX
	MOVSS target+16(FP), X0

	TESTQ BX, BX
	JLE   ret_zero

	XORQ R8, R8

loop:
	CMPQ BX, $1
	JLE  end_loop

	MOVQ BX, R9
	SHRQ $1, R9

	LEAQ  -1(R8)(R9*1), R10
	MOVSS (AX)(R10*4), X1

	LEAQ    (R8)(R9*1), R12
	UCOMISS X0, X1
	CMOVQCS R12, R8

	SUBQ R9, BX
	JMP  loop

end_loop:
	TESTQ BX, BX
	JLE   done

	MOVSS   (AX)(R8*4), X1
	LEAQ    1(R8), R12
	UCOMISS X0, X1
	CMOVQCS R12, R8

done:
	MOVQ R8, ret+24(FP)
	RET

ret_zero:
	MOVQ $0, ret+24(FP)
	RET
