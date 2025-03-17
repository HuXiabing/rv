#
# from data.tokenizer import RISCVTokenizer
# bb=["ld   a5,16(s0)", "ld    a0,0(s9)", "mv   a2,zero", "auipc  a1,21", "addi  a1,a1,576"]
# tokenizer = RISCVTokenizer(max_instr_length=8)
# token = []
# encoded = []
# for i in bb:
#     encoded.append(tokenizer.encode_instruction(i))
#     token.append(tokenizer.tokenize_instruction(i))
# print(encoded)
# print(token)


# from capstone import *
# #创建capstone对象，设置为32位x86架构
# md = Cs(CS_ARCH_RISCV, CS_MODE_RISCV64)
# md.detail = True
# md.skipdata = True
# # if hasattr(md, 'setOption'):  # 检查API是否支持
# #     print('ds')
# #     md.setOption(CS_OPT_DETAIL, CS_OPT_ON)
# #
# #     # 使用CS_OPT_RISCV_NO_ALIASES选项禁用伪指令
# #     # 注意：需要较新版本的Capstone支持
# #     try:
# #         md.setOption(CS_OPT_RISCV_NO_ALIASES, CS_OPT_ON)
# #     except:
# #         print("警告：当前Capstone版本可能不支持RISCV_NO_ALIASES选项")
# #要反汇编的机器码
# # code = b"\x00\x0b\x5e\x17\xe0\x0e\x3e\x03\x00\x0e\x03\x67\x00\x00\x00\x13"
# code = b"\x17\x5e\x0b\x00\x03\x3e\x0e\xe0\x67\x03\x0e\x00\x13\x00\x00\x00"
#
# # 开始反汇编
# for insn in md.disasm(code, 0x10200):
#     print(insn.mnemonic, insn.op_str)
# """
#    10200:	000b5e17          	auipc	t3,0xb5
#    10204:	e00e3e03          	ld	t3,-512(t3) # c5000 <.got.plt+0x10>
#    10208:	000e0367          	jalr	t1,0(t3)
#    1020c:	00000013          	addi	zero,zero,0
# """